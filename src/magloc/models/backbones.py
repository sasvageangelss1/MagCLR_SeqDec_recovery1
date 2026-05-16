from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

# Some CPU-only PyTorch builds can hang on MKLDNN RNN/LSTM kernels.
# Disabling MKLDNN here keeps the backbone comparison portable for code review machines.
if not torch.cuda.is_available():
    torch.backends.mkldnn.enabled = False

from .convnext1d import ConvNeXtLite1D


class RNNEncoder1D(nn.Module):
    """Simple RNN baseline for supervised magnetic-window localization.

    Input shape is (B, C, L). The recurrent module sees the sequence as
    (B, L, C). The final hidden states are projected to the common embedding
    dimension so that all backbones can share the same regression head.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.10,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.rnn = nn.RNN(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            nonlinearity="tanh",
        )
        factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_dim * factor)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * factor, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B,L,C)
        out, _ = self.rnn(x)
        h = out[:, -1]
        return self.proj(self.norm(h))


class LSTMEncoder1D(nn.Module):
    """LSTM baseline for supervised magnetic-window localization."""

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.10,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_dim * factor)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * factor, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.proj(self.norm(h))


class TemporalBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.10):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CNNTCNEncoder1D(nn.Module):
    """CNN + TCN baseline.

    A shallow CNN stem extracts local magnetic patterns; dilated residual TCN
    blocks enlarge the temporal/spatial receptive field. The final pooled
    representation is projected to the common embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 7,
        channels: int = 128,
        embed_dim: int = 256,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.10,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(*[TemporalBlock1D(channels, kernel_size=kernel_size, dilation=int(d), dropout=dropout) for d in dilations])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(channels, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.tcn(x)
        h = self.pool(x).squeeze(-1)
        return self.proj(h)


class ConvNeXtSupervisedEncoder1D(nn.Module):
    """ConvNeXt-Lite-1D supervised baseline without contrastive pretraining."""

    def __init__(
        self,
        in_channels: int = 7,
        embed_dim: int = 256,
        depths: Sequence[int] = (2, 2, 4, 2),
        dims: Sequence[int] = (64, 128, 256, 256),
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.backbone = ConvNeXtLite1D(
            in_channels=in_channels,
            depths=depths,
            dims=dims,
            kernel_size=kernel_size,
            layer_scale_init=layer_scale_init,
        )
        self.proj = nn.Sequential(nn.Linear(self.backbone.out_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class SupervisedLocalizationModel(nn.Module):
    """Backbone + common regression head used by the backbone comparison experiment."""

    def __init__(self, encoder: nn.Module, embed_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.encoder = encoder
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reg_head(self.encoder(x))


def build_backbone_encoder(name: str, cfg: Dict) -> nn.Module:
    """Factory for thesis Table 3.3 backbone comparison."""
    name = name.lower().replace("-", "_")
    model_cfg = cfg.get("model", {})
    cmp_cfg = cfg.get("backbone_compare", {})
    embed_dim = int(model_cfg.get("embed_dim", 256))
    in_channels = int(model_cfg.get("in_channels", 7))
    dropout = float(cmp_cfg.get("dropout", cfg.get("finetune", {}).get("dropout", 0.10)))
    hidden_dim = int(cmp_cfg.get("rnn_hidden_dim", 128))
    num_layers = int(cmp_cfg.get("rnn_num_layers", 2))
    bidirectional = bool(cmp_cfg.get("bidirectional", False))

    if name in {"rnn", "vanilla_rnn"}:
        return RNNEncoder1D(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    if name in {"lstm"}:
        return LSTMEncoder1D(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    if name in {"cnn_tcn", "cnn+tcn", "cnntcn"}:
        return CNNTCNEncoder1D(
            in_channels=in_channels,
            channels=int(cmp_cfg.get("tcn_channels", 128)),
            embed_dim=embed_dim,
            kernel_size=int(cmp_cfg.get("tcn_kernel_size", 5)),
            dilations=tuple(cmp_cfg.get("tcn_dilations", [1, 2, 4, 8])),
            dropout=dropout,
        )
    if name in {"convnext", "convnext_lite", "convnext_lite_1d", "convnext-lite-1d"}:
        return ConvNeXtSupervisedEncoder1D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=tuple(model_cfg.get("depths", [2, 2, 4, 2])),
            dims=tuple(model_cfg.get("dims", [64, 128, 256, 256])),
            kernel_size=int(model_cfg.get("kernel_size", 7)),
            layer_scale_init=float(model_cfg.get("layer_scale_init", 1e-6)),
        )
    raise ValueError(f"Unknown backbone name: {name}")


def canonical_backbone_name(name: str) -> str:
    name = name.lower().replace("-", "_")
    if name in {"rnn", "vanilla_rnn"}:
        return "RNN"
    if name == "lstm":
        return "LSTM"
    if name in {"cnn_tcn", "cnn+tcn", "cnntcn"}:
        return "CNN+TCN"
    if name in {"convnext", "convnext_lite", "convnext_lite_1d", "convnext-lite-1d"}:
        return "ConvNeXt-Lite-1D"
    return name


DEFAULT_BACKBONES: List[str] = ["rnn", "lstm", "cnn_tcn", "convnext_lite_1d"]
