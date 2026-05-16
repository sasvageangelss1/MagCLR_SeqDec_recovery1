from .magclr import MagCLRNet, RegressionHead, ProjectionHead
from .convnext1d import ConvNeXtLite1D

from .backbones import (
    RNNEncoder1D, LSTMEncoder1D, CNNTCNEncoder1D, ConvNeXtSupervisedEncoder1D,
    SupervisedLocalizationModel, build_backbone_encoder, canonical_backbone_name, DEFAULT_BACKBONES,
)
