from __future__ import annotations
import argparse
from magloc.experiments.evaluate import evaluate_regression

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/base.yaml')
p.add_argument('--ckpt', required=True)
p.add_argument('--split', default='test')
p.add_argument('--output-dir', default=None)
args = p.parse_args()
evaluate_regression(args.config, args.ckpt, args.split, args.output_dir)
