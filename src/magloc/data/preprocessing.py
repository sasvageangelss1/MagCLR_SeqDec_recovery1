from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class WindowBatch:
    windows: np.ndarray       # (M,N,3), standardized/resampled magnetic signal only
    labels: np.ndarray        # (M,2), window-end coordinates
    lengths: List[int]        # per trajectory window counts


def zscore_norm(mag: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = mag.mean(axis=0, keepdims=True)
    std = mag.std(axis=0, keepdims=True)
    return ((mag - mean) / (std + eps)).astype(np.float32)


def detrend_ema(mag: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    ema = np.zeros_like(mag, dtype=np.float32)
    ema[0] = mag[0]
    for t in range(1, len(mag)):
        ema[t] = alpha * mag[t] + (1.0 - alpha) * ema[t - 1]
    return (mag - ema).astype(np.float32)


def compute_arc_length(pos: np.ndarray) -> np.ndarray:
    diffs = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float32)


def _make_strictly_increasing(s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = s.astype(np.float64).copy()
    for i in range(1, len(s)):
        if s[i] <= s[i - 1]:
            s[i] = s[i - 1] + eps
    return s.astype(np.float32)


def interp_by_distance(values: np.ndarray, s: np.ndarray, s_hat: np.ndarray, assume_strict: bool = False) -> np.ndarray:
    if not assume_strict:
        s = _make_strictly_increasing(s)
    out = np.stack([np.interp(s_hat, s, values[:, c]) for c in range(values.shape[1])], axis=1)
    return out.astype(np.float32)


def _clean_mag_pos(arr: np.ndarray, zscore: bool = True, detrend: bool = False, ema_alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    mag = arr[:, :3]
    pos = arr[:, 3:5]
    valid = np.isfinite(mag).all(axis=1) & np.isfinite(pos).all(axis=1)
    mag, pos = mag[valid], pos[valid]
    if len(mag) == 0:
        return mag.astype(np.float32), pos.astype(np.float32)
    if detrend:
        mag = detrend_ema(mag, alpha=ema_alpha)
    if zscore:
        mag = zscore_norm(mag)
    return mag.astype(np.float32), pos.astype(np.float32)


def build_equal_distance_windows(
    arr: np.ndarray,
    window_size: int = 128,
    window_length_m: float = 2.0,
    stride_m: float = 1.0,
    zscore: bool = True,
    detrend: bool = False,
    ema_alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one raw trajectory [mx,my,mz,x,y] into equal-distance windows.

    The label is the window-end coordinate, matching the thesis description.
    """
    mag, pos = _clean_mag_pos(arr, zscore=zscore, detrend=detrend, ema_alpha=ema_alpha)
    if len(mag) < 2:
        return np.zeros((0, window_size, 3), np.float32), np.zeros((0, 2), np.float32)
    s = _make_strictly_increasing(compute_arc_length(pos))
    total = float(s[-1])
    if total < window_length_m:
        return np.zeros((0, window_size, 3), np.float32), np.zeros((0, 2), np.float32)
    grid_unit = np.linspace(0.0, window_length_m, window_size, dtype=np.float32)
    windows, labels = [], []
    start = 0.0
    while start + window_length_m <= total + 1e-6:
        s_hat = start + grid_unit
        windows.append(interp_by_distance(mag, s, s_hat, assume_strict=True))
        labels.append(interp_by_distance(pos, s, np.array([s_hat[-1]], dtype=np.float32), assume_strict=True)[0])
        start += stride_m
    if not windows:
        return np.zeros((0, window_size, 3), np.float32), np.zeros((0, 2), np.float32)
    return np.stack(windows).astype(np.float32), np.stack(labels).astype(np.float32)


def build_fixed_time_windows(
    arr: np.ndarray,
    window_size: int = 128,
    time_stride_points: int | None = None,
    window_length_m: float = 2.0,
    stride_m: float = 1.0,
    zscore: bool = True,
    detrend: bool = False,
    ema_alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Traditional fixed-time/fixed-sample window baseline for A1 ablation.

    This does not enforce equal physical distance coverage.  The default stride is
    inferred from the original sampling density and the thesis stride_m, but can
    be overridden by preprocess.time_stride_points in the config.
    """
    mag, pos = _clean_mag_pos(arr, zscore=zscore, detrend=detrend, ema_alpha=ema_alpha)
    if len(mag) < window_size:
        return np.zeros((0, window_size, 3), np.float32), np.zeros((0, 2), np.float32)
    if time_stride_points is None or int(time_stride_points) <= 0:
        s = compute_arc_length(pos)
        total = float(s[-1])
        points_per_meter = (len(mag) - 1) / max(total, 1e-6)
        time_stride_points = max(1, int(round(float(stride_m) * points_per_meter)))
    stride = max(1, int(time_stride_points))
    windows, labels = [], []
    for start in range(0, len(mag) - window_size + 1, stride):
        end = start + window_size
        windows.append(mag[start:end])
        labels.append(pos[end - 1])
    if not windows:
        return np.zeros((0, window_size, 3), np.float32), np.zeros((0, 2), np.float32)
    return np.stack(windows).astype(np.float32), np.stack(labels).astype(np.float32)


def prepare_trajectories(arrays: Iterable[np.ndarray], window_mode: str = "equal_distance", **kwargs) -> WindowBatch:
    all_w, all_y, lengths = [], [], []
    mode = (window_mode or "equal_distance").lower()
    eq_kwargs = dict(kwargs)
    eq_kwargs.pop("time_stride_points", None)
    for arr in arrays:
        if mode in {"equal_distance", "distance", "ed"}:
            w, y = build_equal_distance_windows(arr, **eq_kwargs)
        elif mode in {"fixed_time", "time", "wo_equal_distance", "w/o_equal_distance"}:
            w, y = build_fixed_time_windows(arr, **kwargs)
        else:
            raise ValueError(f"unknown window_mode={window_mode!r}; use equal_distance or fixed_time")
        lengths.append(len(w))
        if len(w):
            all_w.append(w)
            all_y.append(y)
    if not all_w:
        window_size = int(kwargs.get("window_size", 128))
        return WindowBatch(np.zeros((0, window_size, 3), np.float32), np.zeros((0, 2), np.float32), lengths)
    return WindowBatch(np.concatenate(all_w, axis=0), np.concatenate(all_y, axis=0), lengths)


def local_variation_features(seq: np.ndarray, diff_k: int = 1, eps: float = 1e-6, use_local_variation: bool = True) -> np.ndarray:
    """Return 3-axis raw input or 7-channel local-variation enhanced features."""
    seq = np.asarray(seq, dtype=np.float32)
    if not use_local_variation:
        return seq.astype(np.float32)
    diff = np.zeros_like(seq, dtype=np.float32)
    k = max(1, int(diff_k))
    if k < len(seq):
        diff[k:] = seq[k:] - seq[:-k]
    energy = np.sqrt((diff * diff).sum(axis=1, keepdims=True) + eps).astype(np.float32)
    return np.concatenate([seq, diff, energy], axis=1).astype(np.float32)
