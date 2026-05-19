"""
序列解码评估脚本：用预训练编码器对测试集做时序路径解码定位（SeqDec）。
流程：提取编码特征 → 加位移一致性约束 → 跳点抑制 → 置信度加权 → 输出最优轨迹
输出：{seqdec_trajectory, ground_truth} 的 NPZ 文件
"""
from __future__ import annotations
import argparse
from magloc.experiments.evaluate import evaluate_seqdec

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/base.yaml')
p.add_argument('--encoder-ckpt', required=True)
p.add_argument('--split', default='test')
p.add_argument('--output-dir', default=None)
args = p.parse_args()
evaluate_seqdec(args.config, args.encoder_ckpt, args.split, args.output_dir)
