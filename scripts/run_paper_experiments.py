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
p.add_argument("--lodo-configs", default=None, help="Comma-separated LODO config paths. Only used by E3.")
args = p.parse_args()
experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
lodo_configs = [x.strip() for x in args.lodo_configs.split(",") if x.strip()] if args.lodo_configs else None
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
    lodo_configs=lodo_configs,
)
