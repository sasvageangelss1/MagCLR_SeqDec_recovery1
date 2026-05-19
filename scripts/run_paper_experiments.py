"""
一键运行论文全部实验并导出标准格式 CSV 的入口脚本。
支持的实验类型：
  e1          - E1 主干网络对比（RNN/LSTM/CNN+TCN/ConvNeXt 等有监督基线）
  e2          - E2 对比学习预训练有效性验证（Scratch vs Pretrain）
  e3          - E3 下游定位方式与跨设备泛化综合实验（LODO）
  a1          - A1 等空间窗口构造消融实验
  a2          - A2 表征增强机制综合消融（数据增强 / 局部变异特征）
  continuous  - 整体连续定位性能对比（含 WKNN/PDR 基线 + MagCLR 全套方法）
  transition  - 状态转移约束消融（位移一致性 / 跳点抑制）
  all         - 运行上述全部实验
输出：
  paper_error_curves.csv      - 与论文上传格式一致的误差曲线数据
  paper_summary_metrics.csv    - 聚合指标汇总表
  paper_summary_metrics.json   - 同上 JSON 格式
"""
from __future__ import annotations

import argparse

from magloc.experiments.paper_runner import run_paper_experiments

p = argparse.ArgumentParser(description="Run/collect thesis experiments and export one uploaded-format error CSV.")
p.add_argument("--config", default="configs/base.yaml")
p.add_argument("--experiments", default="all", help="Comma-separated: e1,e2,e3,a1,a2,continuous,transition,all")
p.add_argument("--csv", default=None, help="Output error-curve CSV. Columns match the uploaded CSV.")
p.add_argument("--summary-csv", default=None, help="Output aggregate metrics CSV.")
p.add_argument("--scene-label", default=None, help="e.g. 场景1 or 场景2. Auto-inferred if omitted.")
p.add_argument("--scene-code", default=None, help="e.g. S1 or S2. Auto-inferred if omitted.")
p.add_argument("--encoding", default="gbk", help="Default gbk to match the uploaded CSV; use utf-8-sig if preferred.")
p.add_argument("--force", action="store_true", help="Rerun even if checkpoints/results already exist.")
p.add_argument("--collect-only", action="store_true", help="Only collect existing .npz outputs into CSV; do not train/evaluate.")
args = p.parse_args()
experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
run_paper_experiments(
    args.config,
    experiments=experiments,
    csv_path=args.csv,
    summary_csv=args.summary_csv,
    scene_label=args.scene_label,
    scene_code=args.scene_code,
    encoding=args.encoding,
    force=args.force,
    collect_only=args.collect_only,
)
