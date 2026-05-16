from __future__ import annotations

import argparse
from pathlib import Path

from magloc.experiments.paper_runner import collect_existing_results_to_csv

p = argparse.ArgumentParser(description="Collect existing .npz experiment outputs into the paper-format CSV.")
p.add_argument("--config", default="configs/base.yaml")
p.add_argument("--csv", default=None)
p.add_argument("--summary-csv", default=None)
p.add_argument("--scene-label", default=None)
p.add_argument("--scene-code", default=None)
p.add_argument("--encoding", default="gbk")
args = p.parse_args()
collect_existing_results_to_csv(
    args.config,
    csv_path=args.csv,
    summary_csv=args.summary_csv,
    scene_label=args.scene_label,
    scene_code=args.scene_code,
    encoding=args.encoding,
)
