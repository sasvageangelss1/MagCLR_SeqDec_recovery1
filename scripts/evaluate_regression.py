"""
回归评估脚本：用已微调的回归模型在测试集上做定位预测，输出预测结果 NPZ 文件。
输入：训练好的回归模型（回归头 + 编码器）检查点
输出：{predictions, ground_truth} 的 NPZ 文件，供后续误差分析与可视化使用
"""
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
