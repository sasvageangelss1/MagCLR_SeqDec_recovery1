from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext1d import ConvNeXtLite1D


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class MagCLRNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        embed_dim: int = 256,
        proj_dim: int = 128,
        depths=(2, 2, 4, 2),
        dims=(64, 128, 256, 256),
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.backbone = ConvNeXtLite1D(in_channels, depths=depths, dims=dims, kernel_size=kernel_size, layer_scale_init=layer_scale_init)
        self.embed = nn.Linear(self.backbone.out_dim, embed_dim)
        self.proj = ProjectionHead(embed_dim, embed_dim, proj_dim)

    def forward(self, x: torch.Tensor, return_proj: bool = True):
        h = self.embed(self.backbone(x))
        if return_proj:
            return h, self.proj(h)
        return h


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
