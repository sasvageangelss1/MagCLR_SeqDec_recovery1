"""
MagCLR 微调（Fine-tune）模块：基于预训练编码器 + 回归头进行位置定位训练。

两阶段训练模式：
  Stage 1 - 预训练（见 pretrain.py）：通过对比学习学习通用的磁力信号表征
  Stage 2 - 微调（此处）：冻结或解冻编码器，添加回归头，用位置标签直接监督

关键设计：
  - RegressionHead：全连接回归头，输出 (x, y, z) 三维坐标（或二维）
  - 标签归一化：用训练集标签均值/标准差对标签做 Z-score 标准化，加速收敛
  - 损失函数：Huber Loss（对异常值比 L2 更鲁棒）
  - Scratch vs Pretrain：--scratch 时从随机初始化开始训练；否则加载预训练编码器权重
  - 参数冻结策略：默认冻结编码器主体，只训练回归头 + 编码器最后一层_embed

训练流程：
  1. 加载预训练编码器（如指定）并加载训练/验证窗口
  2. 构建 RegressionHead；创建 RegressionWindowDataset（含数据增强）
  3. 每个 epoch：在训练集上前向传播+反向更新，在验证集上评估定位误差
  4. 保存验证集 mean_error 最优的检查点（含编码器+回归头+归一化参数）
  5. 导出 loss_history.json / loss_curve.png（含训练损失 & 验证均方误差曲线）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from magloc.data.datasets import RegressionWindowDataset
from magloc.experiments.common import build_aug, load_windows_for_split, make_model
from magloc.models import RegressionHead
from magloc.eval.metrics import localization_metrics, save_metrics
from magloc.utils import ensure_dir, get_device, load_yaml, set_seed


def _save_loss_history(history: list[dict], out: Path) -> None:
    """保存训练历史（JSON+CSV）并绘制双图：训练损失 & 验证均方误差曲线。"""
    json_path = out / "loss_history.json"
    csv_path = out / "loss_history.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    if history:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
    _plot_loss_curves(history, out)


def _plot_loss_curves(history: list[dict], out: Path) -> None:
    """绘制训练损失（左图）和验证均方误差（右图）随 Epoch 变化的曲线。"""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_mean_error = [h["val_mean_error"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, marker="o", markersize=3, linewidth=1.5, color="#2196F3")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_mean_error, marker="o", markersize=3, linewidth=1.5, color="#FF5722")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean Error (m)")
    axes[1].set_title("Validation Mean Error"); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    plt.close(fig)


def _eval(model, head, loader, device, pos_mean=None, pos_std=None):
    """
    在给定数据集上执行推理，返回预测坐标和真实坐标。

    参数:
        pos_mean / pos_std: 标签归一化的均值和标准差；若提供则将预测反归一化到原始尺度
    返回:
        (predictions, ground_truths): 两个 numpy 数组，形状均为 (N, coord_dim)
    """
    model.eval(); head.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model(x, return_proj=False)  # 提取特征，不经过投影头
            pred = head(h)                    # 回归头输出
            pred = pred.cpu().numpy()
            y_np = y.numpy()
            if pos_mean is not None:
                pred = pred * pos_std + pos_mean  # 反归一化到原始坐标尺度
            preds.append(pred); gts.append(y_np)
    return np.concatenate(preds), np.concatenate(gts)


def run_finetune(
    config_path: str,
    pretrained_ckpt: str | None = None,
    output_dir: str | None = None,
    scratch: bool = False,
) -> Path:
    """
    执行完整的微调训练流程。

    参数:
        config_path:   配置文件路径
        pretrained_ckpt: 预训练编码器检查点路径；若为 None 且 scratch=False 则内部自动查找
        output_dir:    可选，输出目录；默认保存到 {output_root}/{scene_name}/finetune/
        scratch:       若为 True，从随机初始化开始训练，不加载预训练权重

    返回:
        best_path: 验证集 mean_error 最优的回归模型检查点路径
    """
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 2026)))
    # 输出目录：finetune_scratch（从头训练）或 finetune（微调）
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] /
                     ("finetune_scratch" if scratch else "finetune"))

    train_batch, _ = load_windows_for_split(cfg, "train")
    val_batch, _ = load_windows_for_split(cfg, "val")
    print(f"[finetune] train={train_batch.windows.shape}, val={val_batch.windows.shape}")

    device = get_device()
    model = make_model(cfg).to(device)

    # 加载预训练编码器权重（非 Scratch 模式）
    if pretrained_ckpt and not scratch:
        ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model") or ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"[finetune] loaded pretrained {pretrained_ckpt}")

    # RegressionHead：两层全连接网络，输出坐标 (x, y) 或 (x, y, z)
    head = RegressionHead(int(cfg["model"].get("embed_dim", 256)),
                         dropout=float(cfg["finetune"].get("dropout", 0.1))).to(device)

    diff_k = int(cfg["preprocess"].get("diff_k", 1))
    aug = build_aug(cfg)
    # 训练集开启数据增强，验证集关闭（保证评估一致性）
    train_ds = RegressionWindowDataset(
        train_batch.windows, train_batch.labels,
        diff_k=diff_k, augment=True, aug=aug,
        use_local_variation=bool(cfg["preprocess"].get("msfe", True))
    )
    val_ds = RegressionWindowDataset(
        val_batch.windows, val_batch.labels,
        diff_k=diff_k, augment=False, aug=aug,
        use_local_variation=bool(cfg["preprocess"].get("msfe", True))
    )

    ft = cfg["finetune"]
    train_loader = DataLoader(train_ds, batch_size=int(ft.get("batch_size", 128)),
                             shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(ft.get("batch_size", 128)),
                            shuffle=False, num_workers=0)

    # 标签归一化：用训练集均值/标准差对标签做 Z-score 变换
    label_norm = bool(ft.get("label_norm", True))
    pos_mean = train_batch.labels.mean(axis=0).astype(np.float32) if label_norm else None
    pos_std = (train_batch.labels.std(axis=0) + 1e-6).astype(np.float32) if label_norm else None
    pos_mean_t = torch.tensor(pos_mean, device=device) if label_norm else None
    pos_std_t = torch.tensor(pos_std, device=device) if label_norm else None

    # 参数分组：Scratch 时训练全部参数；否则冻结编码器主体，只训练回归头 + 编码器_embed 层
    params = list(model.parameters()) + list(head.parameters()) if scratch \
             else list(head.parameters()) + list(model.embed.parameters())
    opt = torch.optim.AdamW(params,
                            lr=float(ft.get("lr", 1e-3)),
                            weight_decay=float(ft.get("weight_decay", 1e-4)))

    best = float("inf")
    bad = 0
    best_path = out / "regression_best.pth"
    history = []

    for epoch in range(1, int(ft.get("epochs", 80)) + 1):
        model.train(); head.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"finetune {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            # 标签归一化：与推理时保持一致
            yy = (y - pos_mean_t) / pos_std_t if label_norm else y
            # return_proj=False 提取原始特征而非投影特征，用于回归
            pred = head(model(x, return_proj=False))
            # Huber Loss：结合 L1 和 L2 优点，对大误差比纯 L2 更不敏感
            loss = F.smooth_l1_loss(pred, yy, beta=float(ft.get("huber_beta", 1.0)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.item()))

        # 在验证集上评估定位误差
        pred, gt = _eval(model, head, val_loader, device, pos_mean, pos_std)
        metrics = localization_metrics(
            pred, gt,
            jump_threshold_m=float(cfg["evaluation"].get("jump_threshold_m", 2.5))
        )
        avg_loss = float(np.mean(losses))
        print(f"[finetune] epoch={epoch} loss={avg_loss:.6f} "
              f"val_mean={metrics['mean_error']:.4f} p90={metrics['p90_error']:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_mean_error": float(metrics["mean_error"]),
            "val_p90_error": float(metrics["p90_error"]),
            "best": metrics["mean_error"] < best
        })

        # 保存验证集 mean_error 最优的检查点（含编码器、回归头、归一化参数）
        if metrics["mean_error"] < best:
            best = metrics["mean_error"]
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "pos_mean": pos_mean if label_norm else None,
                "pos_std": pos_std if label_norm else None,
                "label_norm": label_norm,
            }, best_path)
            save_metrics(metrics, out / "regression_val_metrics.json")
        else:
            bad += 1

        # 早停机制
        if int(ft.get("early_stop_patience", 0)) and bad >= int(ft.get("early_stop_patience")):
            print("[finetune] early stop")
            break

    _save_loss_history(history, out)
    return best_path
