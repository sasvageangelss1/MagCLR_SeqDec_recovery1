#!/usr/bin/env python
"""CLI entry point for PDR baseline evaluation (IMU-based)."""

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU-only to avoid CUDA crash

import sys as _sys
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_src_dir = _os.path.normpath(_os.path.join(_script_dir, "..", "src"))
if _os.path.isdir(_src_dir) and _src_dir not in _sys.path:
    _sys.path.insert(0, _src_dir)

import matplotlib as _mpl
_mpl.use("Agg")   # headless backend BEFORE any other matplotlib imports

import argparse

from magloc.experiments.evaluate_baselines import evaluate_pdr


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PDR (Pedestrian Dead Reckoning) baseline using IMU sensors"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--split", default="test",
        help="Evaluation split (default: test)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: {output_root}/{scene}/eval_pdr/)"
    )
    parser.add_argument(
        "--step-length", type=float, default=0.65,
        help="Assumed constant step length in metres (default: 0.65)"
    )
    parser.add_argument(
        "--prominence", type=float, default=0.8,
        help="Peak prominence for step detection in standardised units (default: 0.8)"
    )
    parser.add_argument(
        "--min-interval", type=int, default=15,
        help="Minimum raw samples between detected steps (default: 15)"
    )
    parser.add_argument(
        "--heading-window", type=int, default=5,
        help="Window radius for heading estimation -- kept for CLI compat, unused in gyro mode (default: 5)"
    )
    parser.add_argument(
        "--spacing", type=float, default=0.05,
        help="Arc-length resampling resolution in metres -- kept for CLI compat, unused in IMU-PDR (default: 0.05)"
    )
    parser.add_argument(
        "--gyro-lpf-alpha", type=float, default=0.85,
        help="IIR low-pass alpha on gyro Z before integration (default: 0.85, higher = smoother)"
    )
    parser.add_argument(
        "--comp-alpha", type=float, default=1.0,
        help="Complementary-filter gyro weight (default: 1.0 = pure gyro; use <1 to fuse magnetometer)"
    )
    args = parser.parse_args()
    evaluate_pdr(
        config_path=args.config,
        split_name=args.split,
        output_dir=args.output_dir,
        step_length=args.step_length,
        prominence=args.prominence,
        min_interval=args.min_interval,
        heading_window=args.heading_window,
        spacing_m=args.spacing,
        gyro_lpf_alpha=args.gyro_lpf_alpha,
        comp_alpha=args.comp_alpha,
    )


if __name__ == "__main__":
    main()
