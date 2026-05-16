from __future__ import annotations
import argparse
from magloc.experiments.finetune import run_finetune

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/base.yaml')
p.add_argument('--pretrained-ckpt', default=None)
p.add_argument('--output-dir', default=None)
p.add_argument('--scratch', action='store_true')
args = p.parse_args()
run_finetune(args.config, args.pretrained_ckpt, args.output_dir, scratch=args.scratch)
