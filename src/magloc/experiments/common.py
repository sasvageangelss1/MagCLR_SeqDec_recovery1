from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from magloc.data.datasets import AugmentConfig
from magloc.data.io import load_split
from magloc.data.preprocessing import WindowBatch, prepare_trajectories
from magloc.models import MagCLRNet
from magloc.utils import get_device


def build_aug(cfg) -> AugmentConfig:
    a = cfg.get("augmentation", {})
    return AugmentConfig(
        rotation_max_deg=float(a.get("rotation_max_deg", 20.0)),
        noise_sigma=float(a.get("noise_sigma", 0.01)),
        # crop_ratio_min=float(a.get("crop_ratio_min", 0.90)),
        # crop_ratio_max=float(a.get("crop_ratio_max", 1.00)),
        # grid_jitter_prob=float(a.get("grid_jitter_prob", 0.20)),
        # grid_jitter_std=float(a.get("grid_jitter_std", 0.04)),
        # channel_dropout_prob=float(a.get("channel_dropout_prob", 0.0)),
        # channel_shuffle_prob=float(a.get("channel_shuffle_prob", 0.0)),
    )


def preprocess_kwargs(cfg):
    p = cfg["preprocess"]
    return dict(
        window_size=int(p.get("window_size", 128)),
        window_length_m=float(p.get("window_length_m", 2.0)),
        stride_m=float(p.get("stride_m", 1.0)),
        window_mode=str(p.get("window_mode", "equal_distance")),
        time_stride_points=p.get("time_stride_points", None),
        zscore=bool(p.get("zscore", True)),
        detrend=bool(p.get("detrend_ema", False)),
        ema_alpha=float(p.get("ema_alpha", 0.1)),
    )


def load_windows_for_split(cfg, split_name: str) -> tuple[WindowBatch, list[Path]]:
    root = Path(cfg["paths"]["data_root"])
    split_dir_name = cfg["split"].get(f"{split_name}_dir", split_name)
    pattern = cfg["split"].get("file_pattern", "*.npy")
    scene_filter = cfg.get("scene", {}).get("scene_filter") or None
    arrays, files = load_split(root, split_dir_name, pattern=pattern, scene_filter=scene_filter)
    batch = prepare_trajectories(arrays, **preprocess_kwargs(cfg))
    return batch, files


def make_model(cfg) -> MagCLRNet:
    m = cfg["model"]
    return MagCLRNet(
        in_channels=int(m.get("in_channels", 7)),
        embed_dim=int(m.get("embed_dim", 256)),
        proj_dim=int(m.get("proj_dim", 128)),
        depths=tuple(m.get("depths", [2, 2, 4, 2])),
        dims=tuple(m.get("dims", [64, 128, 256, 256])),
        kernel_size=int(m.get("kernel_size", 7)),
        layer_scale_init=float(m.get("layer_scale_init", 1e-6)),
    )


def load_encoder(cfg, ckpt_path: str | Path, strict: bool = False) -> MagCLRNet:
    device = get_device()
    model = make_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model
