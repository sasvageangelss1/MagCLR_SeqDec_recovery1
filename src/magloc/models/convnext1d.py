from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(dim) * layer_scale_init) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1)
        return residual + x


class ConvNeXtLite1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        depths: Sequence[int] = (2, 2, 4, 2),
        dims: Sequence[int] = (64, 128, 256, 256),
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.out_dim = int(dims[-1])
        self.stem = nn.Sequential(nn.Conv1d(in_channels, dims[0], kernel_size=4, stride=4), LayerNorm1d(dims[0]))
        stages = []
        for i, depth in enumerate(depths):
            stages.append(nn.Sequential(*[ConvNeXtBlock1D(dims[i], kernel_size, layer_scale_init) for _ in range(depth)]))
            if i < len(depths) - 1:
                stages.append(nn.Sequential(nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2), LayerNorm1d(dims[i + 1])))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.pool(x).squeeze(-1)
