#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
CONFIG=${1:-configs/base.yaml}

python scripts/train_pretrain.py --config "$CONFIG"
python scripts/train_regression.py --config "$CONFIG" --pretrained-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
python scripts/evaluate_regression.py --config "$CONFIG" --ckpt experiments/scenario_1/finetune/regression_best.pth
python scripts/evaluate_retrieval.py --config "$CONFIG" --encoder-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
python scripts/evaluate_seqdec.py --config "$CONFIG" --encoder-ckpt experiments/scenario_1/pretrain/pretrain_best.pth
