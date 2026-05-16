from __future__ import annotations

import math
import numpy as np


def interp_to_len(seq: np.ndarray, length: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == length:
        return seq
    if len(seq) < 2:
        return np.repeat(seq, length, axis=0)
    src = np.arange(len(seq))
    dst = np.linspace(0, len(seq) - 1, length)
    return np.stack([np.interp(dst, src, seq[:, c]) for c in range(seq.shape[1])], axis=1).astype(np.float32)


def random_end_aligned_crop(seq: np.ndarray, crop_ratio_min: float = 0.90, crop_ratio_max: float = 1.00) -> np.ndarray:
    n = len(seq)
    r = np.random.uniform(crop_ratio_min, crop_ratio_max)
    keep = max(2, min(n, int(round(n * r))))
    start = n - keep
    return interp_to_len(seq[start:], n)


def distance_grid_jitter(seq: np.ndarray, std: float = 0.04) -> np.ndarray:
    n, c = seq.shape
    if n < 3 or std <= 0:
        return seq.astype(np.float32)
    base = np.arange(n, dtype=np.float32)
    noise = np.random.normal(0.0, std, n).astype(np.float32)
    noise[0] = 0.0
    noise[-1] = 0.0
    grid = np.maximum.accumulate(base + noise)
    grid = np.clip(grid, 0.0, n - 1)
    for i in range(1, n):
        if grid[i] <= grid[i - 1]:
            grid[i] = min(n - 1.0, grid[i - 1] + 1e-3)
    return np.stack([np.interp(base, grid, seq[:, j]) for j in range(c)], axis=1).astype(np.float32)


def random_rotation_3d(seq: np.ndarray, max_deg: float = 20.0) -> np.ndarray:
    if max_deg <= 0:
        return seq.astype(np.float32)
    angles = np.deg2rad(np.random.uniform(-max_deg, max_deg, size=3))
    ax, ay, az = angles
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]], dtype=np.float32)
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]], dtype=np.float32)
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    return (seq @ R.T).astype(np.float32)


def jitter(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    if sigma <= 0:
        return seq.astype(np.float32)
    return (seq + np.random.normal(0.0, sigma, size=seq.shape).astype(np.float32)).astype(np.float32)


def maybe_channel_dropout(seq: np.ndarray, p: float = 0.0) -> np.ndarray:
    if p > 0 and np.random.rand() < p:
        x = seq.copy()
        x[:, np.random.randint(seq.shape[1])] = 0.0
        return x
    return seq


def maybe_channel_shuffle(seq: np.ndarray, p: float = 0.0) -> np.ndarray:
    if p > 0 and np.random.rand() < p:
        return seq[:, np.random.permutation(seq.shape[1])].copy()
    return seq
