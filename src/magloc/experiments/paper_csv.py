from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

PAPER_CSV_COLUMNS = ["figure_id", "chapter_section", "scene", "method", "curve_key", "sample_id", "error_m"]
SUMMARY_COLUMNS = [
    "figure_id", "chapter_section", "scene", "method", "curve_key", "count",
    "mean_error", "median_error", "p90_error", "rmse_error", "max_error",
]


def errors_from_pred_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    return np.linalg.norm(pred - gt, axis=1).astype(np.float32)


def read_errors_from_npz(path: str | Path, pred_key: str = "pred", gt_key: str = "gt") -> np.ndarray:
    arr = np.load(path)
    return errors_from_pred_gt(arr[pred_key], arr[gt_key])


def make_error_rows(
    errors: Sequence[float] | np.ndarray,
    figure_id: str,
    chapter_section: str,
    scene: str,
    method: str,
    curve_key: str,
) -> List[Dict[str, object]]:
    e = np.asarray(errors, dtype=np.float32).reshape(-1)
    rows: List[Dict[str, object]] = []
    for i, val in enumerate(e, start=1):
        rows.append({
            "figure_id": figure_id,
            "chapter_section": chapter_section,
            "scene": scene,
            "method": method,
            "curve_key": curve_key,
            "sample_id": i,
            "error_m": float(val),
        })
    return rows


def write_error_csv(rows: Iterable[Dict[str, object]], csv_path: str | Path, encoding: str = "gbk") -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding=encoding, errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PAPER_CSV_COLUMNS})
    return csv_path


def append_error_rows(rows: Iterable[Dict[str, object]], csv_path: str | Path, encoding: str = "gbk") -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding=encoding, errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PAPER_CSV_COLUMNS})
    return csv_path


def load_error_csv(csv_path: str | Path, encoding: str = "gbk") -> List[Dict[str, object]]:
    with open(csv_path, "r", encoding=encoding, errors="replace") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[float]] = {}
    meta: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        key = (row["figure_id"], row["chapter_section"], row["scene"], row["method"], row["curve_key"])
        groups.setdefault(key, []).append(float(row["error_m"]))
        meta[key] = {k: row[k] for k in ["figure_id", "chapter_section", "scene", "method", "curve_key"]}
    out: List[Dict[str, object]] = []
    for key, vals in groups.items():
        e = np.asarray(vals, dtype=np.float64)
        item = dict(meta[key])
        item.update({
            "count": int(len(e)),
            "mean_error": float(np.mean(e)),
            "median_error": float(np.median(e)),
            "p90_error": float(np.percentile(e, 90)),
            "rmse_error": float(np.sqrt(np.mean(e ** 2))),
            "max_error": float(np.max(e)),
        })
        out.append(item)
    return out


def write_summary_csv(summary_rows: Iterable[Dict[str, object]], path: str | Path, encoding: str = "gbk") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding=encoding, errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})
    return path


def write_summary_json(summary_rows: Iterable[Dict[str, object]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(summary_rows), f, ensure_ascii=False, indent=2)
    return path
