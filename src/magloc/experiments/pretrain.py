"""
MagCLR 对比学习预训练模块。

核心思想：同一物理位置在不同时间采集（或经不同增强变换）的两个窗口，
应在编码器输出的特征空间中彼此靠近；不同位置的窗口则应彼此远离。
采用 InfoNCE（Noise Contrastive Estimation）作为对比损失函数。

训练流程：
  1. 加载训练/验证时间窗口数据
  2. 对每个窗口生成两个增强视图（v1, v2）
  3. 用编码器分别提取 v1 和 v2 的投影特征 z1, z2
  4. 计算 info_nce(z1, z2) 对比损失并反向更新编码器
  5. 监控训练损失，保存验证集上最优的模型检查点
  6. 训练完成后导出 loss_history.json / loss_curve.png
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from magloc.data.datasets import ContrastiveWindowDataset
from magloc.experiments.common import build_aug, load_windows_for_split, make_model
from magloc.train.losses import info_nce
from magloc.utils import ensure_dir, get_device, load_yaml, set_seed


def _save_loss_history(history: list[dict], out: Path) -> None:
    """将训练历史保存为 JSON、CSV，并绘制损失曲线图。"""
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
    """绘制训练损失随 Epoch 变化的曲线，保存为 loss_curve.png。"""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", markersize=3, linewidth=1.5, label="train_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    plt.close(fig)


def run_pretrain(config_path: str, output_dir: str | None = None) -> Path:
    """
    执行完整的对比学习预训练流程。

    参数:
        config_path: 配置文件路径（YAML），包含数据路径、模型结构、训练超参数等
        output_dir:  可选，输出目录；若不指定则默认保存到 {output_root}/{scene_name}/pretrain/

    返回:
        best_path: 验证集上训练损失最低的模型检查点路径
    """
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 2026)))  # 固定随机种子，确保可复现
    # 确定输出目录：默认 <output_root>/<scene_name>/pretrain/
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "pretrain")

    # 加载训练/验证时间窗口数据（未增强的原始序列窗口）
    train_batch, _ = load_windows_for_split(cfg, "train")
    val_batch, _ = load_windows_for_split(cfg, "val")
    print(f"[pretrain] train windows={train_batch.windows.shape}, val={val_batch.windows.shape}")

    # 构建数据增强器（随机旋转、加噪声等）
    aug = build_aug(cfg)
    # diff_k: 差分窗口的阶数（默认1阶差分，捕捉一阶变化趋势）
    diff_k = int(cfg["preprocess"].get("diff_k", 1))
    # ContrastiveWindowDataset 对每个窗口生成两个增强视图，并加入局部变异特征（MSFE）
    ds = ContrastiveWindowDataset(
        train_batch.windows, train_batch.labels,
        diff_k=diff_k, aug=aug,
        use_local_variation=bool(cfg["preprocess"].get("msfe", True))
    )

    tr = cfg["pretrain"]
    # DataLoader 负责批量加载和打乱；drop_last=True 保证每 batch 大小一致
    loader = DataLoader(
        ds,
        batch_size=int(tr.get("batch_size", 128)),
        shuffle=True,
        drop_last=True,
        num_workers=int(tr.get("num_workers", 0))
    )

    device = get_device()
    model = make_model(cfg).to(device)  # 初始化编码器模型

    # AdamW 优化器，默认学习率 1e-3，权重衰减 1e-4
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr.get("lr", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-4))
    )

    best = float("inf")   # 记录最低训练损失
    bad = 0                # 连续未改善的 epoch 数（用于早停）
    best_path = out / "pretrain_best.pth"
    history = []

    for epoch in range(1, int(tr.get("epochs", 100)) + 1):
        model.train()
        losses = []
        # 每个 batch: v1 和 v2 是同一窗口的两个增强视图
        for v1, v2, _ in tqdm(loader, desc=f"pretrain {epoch}", leave=False):
            v1, v2 = v1.to(device), v2.to(device)
            # 编码器返回 (投影头输出, 投影特征 z)；InfoNCE 在 z 上计算
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = info_nce(z1, z2, temperature=float(tr.get("temperature", 0.1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        avg = float(np.mean(losses)) if losses else float("inf")
        print(f"[pretrain] epoch={epoch} loss={avg:.6f}")
        history.append({"epoch": epoch, "train_loss": avg, "best": avg < best})

        # 保存验证集上损失最低的模型（此处以训练损失近似，实际应用中可加验证损失）
        if avg < best:
            best = avg
            bad = 0
            torch.save(model.state_dict(), best_path)
            print(f"[pretrain] saved {best_path}")
        else:
            bad += 1

        # 每隔 save_every 个 epoch 保存一次中间检查点
        if epoch % int(tr.get("save_every", 10)) == 0:
            torch.save(model.state_dict(), out / f"pretrain_epoch_{epoch:03d}.pth")

        # 早停：若连续 bad >= early_stop_patience 个 epoch 未改善则提前结束
        if int(tr.get("early_stop_patience", 0)) and bad >= int(tr.get("early_stop_patience")):
            print("[pretrain] early stop")
            break

    _save_loss_history(history, out)
    return best_path
