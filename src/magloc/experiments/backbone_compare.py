from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from magloc.data.datasets import RegressionWindowDataset
from magloc.eval.metrics import localization_metrics, save_metrics
from magloc.experiments.common import build_aug, load_windows_for_split
from magloc.models import DEFAULT_BACKBONES, SupervisedLocalizationModel, build_backbone_encoder, canonical_backbone_name
from magloc.utils import ensure_dir, get_device, load_yaml, save_yaml, set_seed


def _make_loader(batch, cfg: Dict, train: bool, batch_size: int):
    diff_k = int(cfg["preprocess"].get("diff_k", 1))
    aug = build_aug(cfg)
    cmp_cfg = cfg.get("backbone_compare", {})
    use_aug = bool(cmp_cfg.get("train_augment", False)) if train else False
    ds = RegressionWindowDataset(batch.windows, batch.labels, diff_k=diff_k, augment=use_aug, aug=aug, use_local_variation=bool(cfg["preprocess"].get("msfe", True)))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=int(cmp_cfg.get("num_workers", 0)))


@torch.no_grad()
def _evaluate(model, loader, device, pos_mean=None, pos_std=None, jump_threshold_m: float = 2.5):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        pred = model(x.to(device)).cpu().numpy()
        if pos_mean is not None:
            pred = pred * np.asarray(pos_std, dtype=np.float32) + np.asarray(pos_mean, dtype=np.float32)
        preds.append(pred.astype(np.float32))
        gts.append(y.numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    metrics = localization_metrics(pred, gt, jump_threshold_m=jump_threshold_m)
    return pred, gt, metrics


def _write_summary(rows: List[Dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "summary.json"
    csv_path = out / "summary.csv"
    md_path = out / "summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    fieldnames = ["backbone", "best_epoch", "val_mean_error", "test_mean_error", "test_median_error", "test_p90_error", "test_jump_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| 网络类型 | 最佳轮次 | 验证平均误差 | 测试平均误差 | 测试中位误差 | 测试P90误差 | 跳变比例 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['backbone']} | {row['best_epoch']} | "
                f"{row['val_mean_error']:.4f} | {row['test_mean_error']:.4f} | "
                f"{row['test_median_error']:.4f} | {row['test_p90_error']:.4f} | {row['test_jump_ratio']:.4f} |\n"
            )


def _train_one_backbone(
    backbone_key: str,
    cfg: Dict,
    train_batch,
    val_batch,
    test_batch,
    out_root: Path,
    override_epochs: int | None = None,
) -> Dict:
    cmp_cfg = cfg.get("backbone_compare", {})
    ft_cfg = cfg.get("finetune", {})
    device = get_device()
    display_name = canonical_backbone_name(backbone_key)
    safe_name = display_name.lower().replace("+", "_").replace("-", "_").replace(" ", "_")
    out = ensure_dir(out_root / safe_name)

    batch_size = int(cmp_cfg.get("batch_size", ft_cfg.get("batch_size", 128)))
    epochs = int(override_epochs or cmp_cfg.get("epochs", ft_cfg.get("epochs", 80)))
    lr = float(cmp_cfg.get("lr", ft_cfg.get("lr", 1e-3)))
    wd = float(cmp_cfg.get("weight_decay", ft_cfg.get("weight_decay", 1e-4)))
    huber_beta = float(cmp_cfg.get("huber_beta", ft_cfg.get("huber_beta", 1.0)))
    patience = int(cmp_cfg.get("early_stop_patience", ft_cfg.get("early_stop_patience", 20)))
    grad_clip = float(cmp_cfg.get("grad_clip", 1.0))
    label_norm = bool(cmp_cfg.get("label_norm", ft_cfg.get("label_norm", True)))
    jump_threshold = float(cfg.get("evaluation", {}).get("jump_threshold_m", 2.5))

    train_loader = _make_loader(train_batch, cfg, train=True, batch_size=batch_size)
    val_loader = _make_loader(val_batch, cfg, train=False, batch_size=batch_size)
    test_loader = _make_loader(test_batch, cfg, train=False, batch_size=batch_size)

    pos_mean = train_batch.labels.mean(axis=0).astype(np.float32) if label_norm else None
    pos_std = (train_batch.labels.std(axis=0) + 1e-6).astype(np.float32) if label_norm else None
    pos_mean_t = torch.tensor(pos_mean, device=device) if label_norm else None
    pos_std_t = torch.tensor(pos_std, device=device) if label_norm else None

    encoder = build_backbone_encoder(backbone_key, cfg)
    model = SupervisedLocalizationModel(
        encoder,
        embed_dim=int(cfg.get("model", {}).get("embed_dim", 256)),
        hidden_dim=int(cmp_cfg.get("reg_hidden_dim", 128)),
        dropout=float(cmp_cfg.get("dropout", ft_cfg.get("dropout", 0.10))),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(cmp_cfg.get("min_lr", 1e-5)))

    best = float("inf")
    best_epoch = 0
    bad = 0
    best_path = out / "best.pth"
    history = []
    print(f"[backbone] {display_name}: train={train_batch.windows.shape}, val={val_batch.windows.shape}, test={test_batch.windows.shape}")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"{display_name} {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            target = (y - pos_mean_t) / pos_std_t if label_norm else y
            pred = model(x)
            loss = F.smooth_l1_loss(pred, target, beta=huber_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        scheduler.step()
        _, _, val_metrics = _evaluate(model, val_loader, device, pos_mean, pos_std, jump_threshold)
        item = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": float(v) for k, v in val_metrics.items()}}
        history.append(item)
        print(
            f"[backbone] {display_name} epoch={epoch} "
            f"loss={item['train_loss']:.6f} val_mean={val_metrics['mean_error']:.4f} "
            f"val_p90={val_metrics['p90_error']:.4f}"
        )
        if val_metrics["mean_error"] < best:
            best = float(val_metrics["mean_error"])
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "pos_mean": pos_mean,
                    "pos_std": pos_std,
                },
                best_path,
            )
            save_metrics(val_metrics, out / "val_metrics.json")
        else:
            bad += 1
        if patience > 0 and bad >= patience:
            print(f"[backbone] {display_name} early stop at epoch {epoch}")
            break

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    pred, gt, test_metrics = _evaluate(model, test_loader, device, ckpt.get("pos_mean"), ckpt.get("pos_std"), jump_threshold)
    save_metrics(test_metrics, out / "test_metrics.json")
    np.savez_compressed(out / "test_preds.npz", pred=pred, gt=gt)

    row = {
        "backbone": display_name,
        "best_epoch": int(best_epoch),
        "checkpoint": str(best_path),
        "val_mean_error": float(best),
        "test_mean_error": float(test_metrics["mean_error"]),
        "test_median_error": float(test_metrics["median_error"]),
        "test_p90_error": float(test_metrics["p90_error"]),
        "test_jump_ratio": float(test_metrics.get("jump_ratio", 0.0)),
    }
    print(f"[backbone] {display_name} TEST {row}")
    return row


def run_backbone_compare(
    config_path: str,
    backbones: Sequence[str] | None = None,
    output_dir: str | None = None,
    epochs: int | None = None,
) -> Path:
    cfg = load_yaml(config_path)
    # CPU-only review machines may be dramatically slower with many OpenMP threads.
    if not torch.cuda.is_available():
        torch.set_num_threads(int(cfg.get("backbone_compare", {}).get("torch_num_threads", 1)))
    set_seed(int(cfg.get("seed", 2026)))
    cmp_cfg = cfg.setdefault("backbone_compare", {})
    selected = list(backbones or cmp_cfg.get("backbones", DEFAULT_BACKBONES))
    if not selected:
        selected = list(DEFAULT_BACKBONES)

    out_root = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "backbone_compare")
    save_yaml(cfg, out_root / "used_config.yaml")

    train_batch, train_files = load_windows_for_split(cfg, "train")
    val_batch, val_files = load_windows_for_split(cfg, "val")
    test_batch, test_files = load_windows_for_split(cfg, "test")
    meta = {
        "train_windows": int(len(train_batch.windows)),
        "val_windows": int(len(val_batch.windows)),
        "test_windows": int(len(test_batch.windows)),
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "test_files": [str(p) for p in test_files],
        "backbones": selected,
    }
    with open(out_root / "data_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    rows: List[Dict] = []
    for i, b in enumerate(selected):
        # Re-seed each model deterministically but differently so repeated runs are stable.
        set_seed(int(cfg.get("seed", 2026)) + i)
        rows.append(_train_one_backbone(b, cfg, train_batch, val_batch, test_batch, out_root, override_epochs=epochs))
        _write_summary(rows, out_root)
    print(f"[backbone] summary saved to {out_root / 'summary.md'}")
    return out_root
