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
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 2026)))
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "pretrain")
    train_batch, _ = load_windows_for_split(cfg, "train")
    val_batch, _ = load_windows_for_split(cfg, "val")
    print(f"[pretrain] train windows={train_batch.windows.shape}, val={val_batch.windows.shape}")
    aug = build_aug(cfg)
    diff_k = int(cfg["preprocess"].get("diff_k", 1))
    ds = ContrastiveWindowDataset(train_batch.windows, train_batch.labels, diff_k=diff_k, aug=aug, use_local_variation=bool(cfg["preprocess"].get("msfe", True)))
    tr = cfg["pretrain"]
    loader = DataLoader(ds, batch_size=int(tr.get("batch_size", 128)), shuffle=True, drop_last=True, num_workers=int(tr.get("num_workers", 0)))
    device = get_device()
    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr.get("lr", 1e-3)), weight_decay=float(tr.get("weight_decay", 1e-4)))
    best = float("inf")
    bad = 0
    best_path = out / "pretrain_best.pth"
    history = []
    for epoch in range(1, int(tr.get("epochs", 100)) + 1):
        model.train()
        losses = []
        for v1, v2, _ in tqdm(loader, desc=f"pretrain {epoch}", leave=False):
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = info_nce(z1, z2, temperature=float(tr.get("temperature", 0.1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        avg = float(np.mean(losses)) if losses else float("inf")
        print(f"[pretrain] epoch={epoch} loss={avg:.6f}")
        history.append({"epoch": epoch, "train_loss": avg, "best": avg < best})
        if avg < best:
            best = avg
            bad = 0
            torch.save(model.state_dict(), best_path)
            print(f"[pretrain] saved {best_path}")
        else:
            bad += 1
        if epoch % int(tr.get("save_every", 10)) == 0:
            torch.save(model.state_dict(), out / f"pretrain_epoch_{epoch:03d}.pth")
        if int(tr.get("early_stop_patience", 0)) and bad >= int(tr.get("early_stop_patience")):
            print("[pretrain] early stop")
            break
    _save_loss_history(history, out)
    return best_path
