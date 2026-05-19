"""
检索评估脚本：用预训练编码器对测试集做 Top-K 候选位置检索。
流程：提取编码特征 → 用 FAISS 建立索引 → 为每个查询找最近邻候选
输出：{candidates, ground_truth} 的 NPZ 文件，包含每帧的检索误差
"""
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
