from __future__ import annotations
import argparse
from magloc.experiments.evaluate import evaluate_retrieval

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/base.yaml')
p.add_argument('--encoder-ckpt', required=True)
p.add_argument('--split', default='test')
p.add_argument('--output-dir', default=None)
args = p.parse_args()
evaluate_retrieval(args.config, args.encoder_ckpt, args.split, args.output_dir)
