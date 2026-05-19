"""
MagCLR 预训练脚本：使用对比学习（InfoNCE）训练编码器。
输入：一组磁力计/惯性传感器时间窗口
目标：使同一位置的不同增强视图在特征空间中靠近，不同位置的距离拉远
"""
from __future__ import annotations
import argparse
from magloc.experiments.pretrain import run_pretrain

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/base.yaml')
p.add_argument('--output-dir', default=None)
args = p.parse_args()
run_pretrain(args.config, args.output_dir)
