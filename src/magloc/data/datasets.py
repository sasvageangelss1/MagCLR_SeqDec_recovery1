from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import distance_grid_jitter, jitter, maybe_channel_dropout, maybe_channel_shuffle, random_end_aligned_crop, random_rotation_3d
from .preprocessing import local_variation_features


@dataclass
class AugmentConfig:
    rotation_max_deg: float = 20.0
    noise_sigma: float = 0.01
    crop_ratio_min: float = 0.90
    crop_ratio_max: float = 1.00
    grid_jitter_prob: float = 0.20
    grid_jitter_std: float = 0.04
    channel_dropout_prob: float = 0.0
    channel_shuffle_prob: float = 0.0


class ContrastiveWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None, diff_k: int = 1, aug: AugmentConfig = AugmentConfig(), use_local_variation: bool = True):
        self.windows = np.asarray(windows, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.diff_k = diff_k
        self.aug = aug
        self.use_local_variation = use_local_variation

    def __len__(self) -> int:
        return len(self.windows)

    def _view(self, x: np.ndarray) -> np.ndarray:
        x = random_end_aligned_crop(x, self.aug.crop_ratio_min, self.aug.crop_ratio_max)
        if np.random.rand() < self.aug.grid_jitter_prob:
            x = distance_grid_jitter(x, self.aug.grid_jitter_std)
        x = random_rotation_3d(x, self.aug.rotation_max_deg)
        x = jitter(x, self.aug.noise_sigma)
        x = maybe_channel_dropout(x, self.aug.channel_dropout_prob)
        x = maybe_channel_shuffle(x, self.aug.channel_shuffle_prob)
        feat = local_variation_features(x, diff_k=self.diff_k, use_local_variation=self.use_local_variation)
        return feat.T.astype(np.float32)  # (7,N)

    def __getitem__(self, idx: int):
        x = self.windows[idx]
        v1 = torch.from_numpy(self._view(x))
        v2 = torch.from_numpy(self._view(x))
        if self.labels is None:
            return v1, v2, torch.empty(0)
        return v1, v2, torch.from_numpy(self.labels[idx]).float()


class RegressionWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray, diff_k: int = 1, augment: bool = False, aug: AugmentConfig = AugmentConfig(), use_local_variation: bool = True):
        self.windows = np.asarray(windows, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.diff_k = diff_k
        self.augment = augment
        self.aug = aug
        self.use_local_variation = use_local_variation

    def __len__(self) -> int:
        return len(self.windows)

    def _make(self, x: np.ndarray) -> np.ndarray:
        if self.augment:
            x = random_end_aligned_crop(x, self.aug.crop_ratio_min, self.aug.crop_ratio_max)
            if np.random.rand() < self.aug.grid_jitter_prob:
                x = distance_grid_jitter(x, self.aug.grid_jitter_std)
            x = random_rotation_3d(x, min(self.aug.rotation_max_deg, 10.0))
            x = jitter(x, min(self.aug.noise_sigma, 0.006))
        return local_variation_features(x, diff_k=self.diff_k, use_local_variation=self.use_local_variation).T.astype(np.float32)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self._make(self.windows[idx])), torch.from_numpy(self.labels[idx]).float()
