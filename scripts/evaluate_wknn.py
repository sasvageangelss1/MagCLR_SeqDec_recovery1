#!/usr/bin/env python
"""CLI entry point for WKNN baseline evaluation."""

import argparse

from magloc.experiments.evaluate_baselines import evaluate_wknn


def main():
    parser = argparse.ArgumentParser(description="Evaluate WKNN fingerprint baseline")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--split", default="test",
        help="Evaluation split (default: test)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: {output_root}/{scene}/eval_wknn/)"
    )
    parser.add_argument(
        "-k", type=int, default=5,
        help="Number of nearest neighbors (default: 5)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.30,
        help="Softmax temperature for position fusion (default: 0.30)"
    )
    args = parser.parse_args()
    evaluate_wknn(
        config_path=args.config,
        split_name=args.split,
        output_dir=args.output_dir,
        k=args.k,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
