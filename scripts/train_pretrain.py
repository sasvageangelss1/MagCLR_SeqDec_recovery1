from __future__ import annotations
import argparse
from magloc.experiments.pretrain import run_pretrain

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/base.yaml')
p.add_argument('--output-dir', default=None)
args = p.parse_args()
run_pretrain(args.config, args.output_dir)
