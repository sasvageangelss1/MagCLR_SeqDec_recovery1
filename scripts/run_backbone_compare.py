from __future__ import annotations

import argparse

from magloc.experiments.backbone_compare import run_backbone_compare

p = argparse.ArgumentParser(description="Run thesis E1 backbone comparison: RNN / LSTM / CNN+TCN / ConvNeXt-Lite-1D.")
p.add_argument("--config", default="configs/base.yaml")
p.add_argument("--backbones", default=None, help="Comma-separated list, e.g. rnn,lstm,cnn_tcn,convnext_lite_1d")
p.add_argument("--output-dir", default=None)
p.add_argument("--epochs", type=int, default=None, help="Override backbone_compare.epochs for quick checks")
args = p.parse_args()
backbones = [x.strip() for x in args.backbones.split(",") if x.strip()] if args.backbones else None
run_backbone_compare(args.config, backbones=backbones, output_dir=args.output_dir, epochs=args.epochs)
