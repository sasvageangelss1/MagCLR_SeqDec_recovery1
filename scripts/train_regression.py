"""
MagCLR 微调脚本：基于预训练编码器 + 回归头，训练位置预测模型。
输入：预训练好的编码器权重（可选） + 带位置标签的训练数据窗口
目标：用回归损失（Huber Loss）让编码器学习将磁力信号映射到真实坐标
若不传入 --pretrained-ckpt 则从头训练（Scratch），不加 --scratch 则默认使用预训练权重微调
"""
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
