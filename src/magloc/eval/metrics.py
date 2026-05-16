from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def localization_metrics(pred: np.ndarray, gt: np.ndarray, jump_threshold_m: float | None = None) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    e = np.linalg.norm(pred - gt, axis=1)
    out = {
        "mean_error": float(np.mean(e)),
        "median_error": float(np.median(e)),
        "p90_error": float(np.percentile(e, 90)),
        "rmse_l2": float(np.sqrt(np.mean(e ** 2))),
        "mae_x": float(np.mean(np.abs(pred[:, 0] - gt[:, 0]))),
        "mae_y": float(np.mean(np.abs(pred[:, 1] - gt[:, 1]))),
        "count": int(len(e)),
    }
    if jump_threshold_m is not None and len(pred) > 1:
        steps = np.linalg.norm(pred[1:] - pred[:-1], axis=1)
        out["jump_ratio"] = float(np.mean(steps > jump_threshold_m))
        out["max_step"] = float(np.max(steps))
    return out


def save_metrics(metrics: Dict[str, float], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
