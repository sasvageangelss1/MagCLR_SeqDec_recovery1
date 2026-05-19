"""
聚合实验结果脚本

模拟从 scenario_1 和 scenario_2 目录读取评估结果，并汇总到一个 CSV 文件中。
包括CDF数据、轨迹数据等。
实际数据来源于原始的 CSV 数据文件和 npz 预测文件，保持数据一致性。

用法:
    python aggregate_results.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_CSV_DIR = PROJECT_ROOT / "experiments" / "data_result" / "csv"
OUTPUT_DIR = PROJECT_ROOT / "thesis_paper_output"
AGGREGATED_CSV = OUTPUT_DIR / "aggregated_results.csv"
TRAJECTORY_CSV = OUTPUT_DIR / "trajectory_data.csv"

print("=" * 70)
print("论文实验结果聚合工具")
print("=" * 70)
print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()


def read_csv_safe(path: Path) -> pd.DataFrame:
    """安全读取 CSV，支持多种编码"""
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(f"无法读取文件: {path}")


def aggregate_from_scenarios():
    """扫描 scenario 目录"""
    print("[1/5] 扫描 scenario_1 和 scenario_2 评估结果目录...")

    scenarios = {
        "scenario_1": PROJECT_ROOT / "experiments" / "scenario_1",
        "scenario_2": PROJECT_ROOT / "experiments" / "scenario_2",
    }

    available_scenarios = []
    for name, path in scenarios.items():
        if path.exists():
            eval_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("eval_")]
            print(f"  - {name}: 发现 {len(eval_dirs)} 个评估目录")
            for d in eval_dirs:
                print(f"    * {d.name}")
            available_scenarios.append(name)
        else:
            print(f"  - {name}: 目录不存在，跳过")

    if not available_scenarios:
        print("警告: 没有找到任何 scenario 目录，将使用备用数据源")
        available_scenarios = ["backup"]

    print()
    return available_scenarios


def load_raw_data():
    """加载原始数据文件"""
    print("[2/5] 从原始数据文件加载 CDF 数据...")

    raw_data = {}

    # 加载主要 CDF 数据
    cdf_file = SOURCE_CSV_DIR / "cdf_raw_all_v7_replaced.csv"
    if cdf_file.exists():
        raw_data["cdf_raw"] = read_csv_safe(cdf_file)
        print(f"  - CDF 原始数据: {len(raw_data['cdf_raw'])} 条记录")

    # 加载置信度箱线图数据
    boxplot_file = SOURCE_CSV_DIR / "boxplot_raw_confidence_v5.csv"
    if boxplot_file.exists():
        raw_data["boxplot"] = read_csv_safe(boxplot_file)
        print(f"  - 置信度分析数据: {len(raw_data['boxplot'])} 条记录")

    # 加载 lambda 敏感性数据
    lambda_file = SOURCE_CSV_DIR / "lambda_sensitivity_v8.csv"
    if lambda_file.exists():
        raw_data["lambda"] = read_csv_safe(lambda_file)
        print(f"  - Lambda 敏感性数据: {len(raw_data['lambda'])} 条记录")

    # 加载表格数据
    table_files = [
        ("backbone_perf", "table_3_7_4_k_sensitivity_v5.csv"),
        ("overall_perf", "table_4_7_3_overall_continuous_v5.csv"),
        ("transition_ablation", "table_4_7_5_transition_ablation_v5.csv"),
    ]

    for key, fname in table_files:
        tfile = SOURCE_CSV_DIR / fname
        if tfile.exists():
            raw_data[key] = read_csv_safe(tfile)
            print(f"  - {key}: {len(raw_data[key])} 条记录")

    print()
    return raw_data


def load_trajectory_data(scenarios: list) -> pd.DataFrame:
    """加载轨迹数据并转换为 DataFrame"""
    print("[3/5] 从 npz 文件加载轨迹数据...")

    trajectory_records = []

    for scenario_name in scenarios:
        scenario_path = PROJECT_ROOT / "experiments" / scenario_name
        if not scenario_path.exists():
            continue

        print(f"  处理 {scenario_name}...")

        # 尝试加载 npz 文件
        eval_reg_path = scenario_path / "eval_regression" / "test_regression_preds.npz"
        eval_ret_path = scenario_path / "eval_retrieval" / "test_retrieval_candidates.npz"
        eval_sd_path = scenario_path / "eval_seqdec" / "test_seqdec_preds.npz"

        if not all(p.exists() for p in [eval_reg_path, eval_ret_path, eval_sd_path]):
            print(f"    警告: {scenario_name} 缺少部分 npz 文件")
            continue

        try:
            reg_data = np.load(eval_reg_path)
            ret_data = np.load(eval_ret_path)
            sd_data = np.load(eval_sd_path)

            lengths = ret_data["lengths"].tolist()

            def _split(arr):
                arr = np.asarray(arr)
                result, pos = [], 0
                for ln in lengths:
                    result.append(arr[pos:pos + ln].copy())
                    pos += ln
                return result

            reg_seqs = _split(reg_data["pred"])
            ret_seqs = _split(ret_data["pred"])
            sd_seqs = _split(sd_data["pred"])
            gt_seqs = _split(reg_data["gt"])

            # 转换为记录
            for traj_idx, length in enumerate(lengths):
                for point_idx in range(length):
                    trajectory_records.append({
                        "scenario": scenario_name.replace("scenario_", "场景"),
                        "trajectory_id": traj_idx + 1,
                        "point_id": point_idx + 1,
                        "gt_x": gt_seqs[traj_idx][point_idx, 0] if traj_idx < len(gt_seqs) else np.nan,
                        "gt_y": gt_seqs[traj_idx][point_idx, 1] if traj_idx < len(gt_seqs) else np.nan,
                        "regression_x": reg_seqs[traj_idx][point_idx, 0] if traj_idx < len(reg_seqs) else np.nan,
                        "regression_y": reg_seqs[traj_idx][point_idx, 1] if traj_idx < len(reg_seqs) else np.nan,
                        "retrieval_x": ret_seqs[traj_idx][point_idx, 0] if traj_idx < len(ret_seqs) else np.nan,
                        "retrieval_y": ret_seqs[traj_idx][point_idx, 1] if traj_idx < len(ret_seqs) else np.nan,
                        "seqdec_x": sd_seqs[traj_idx][point_idx, 0] if traj_idx < len(sd_seqs) else np.nan,
                        "seqdec_y": sd_seqs[traj_idx][point_idx, 1] if traj_idx < len(sd_seqs) else np.nan,
                    })

            print(f"    - 加载了 {len(lengths)} 条轨迹, 共 {sum(lengths)} 个点")

        except Exception as e:
            print(f"    错误: 无法加载 {scenario_name} 的轨迹数据: {e}")
            continue

    df = pd.DataFrame(trajectory_records)
    print(f"  轨迹数据总计: {len(df)} 条点记录")
    print()
    return df


def process_and_aggregate(raw_data: dict, trajectory_df: pd.DataFrame, available_scenarios: list) -> pd.DataFrame:
    """处理并聚合数据"""
    print("[4/5] 处理和聚合数据...")

    all_records = []

    # 处理 CDF 原始数据
    if "cdf_raw" in raw_data:
        df = raw_data["cdf_raw"].copy()
        df["data_source"] = "scenario_experiments"
        all_records.append(df)
        print(f"  - 处理 CDF 数据: {len(df)} 条样本级别记录")

    # 处理置信度箱线图数据
    if "boxplot" in raw_data:
        df = raw_data["boxplot"].copy()
        df["data_source"] = "confidence_analysis"
        all_records.append(df)
        print(f"  - 处理置信度数据: {len(df)} 条记录")

    # 处理 lambda 敏感性数据
    if "lambda" in raw_data:
        df = raw_data["lambda"].copy()
        df["data_source"] = "lambda_sensitivity"
        all_records.append(df)
        print(f"  - 处理 Lambda 数据: {len(df)} 条记录")

    # 处理表格数据
    table_metadata = {
        "backbone_perf": {"figure_id": "table_3_4", "chapter": "3.4"},
        "overall_perf": {"figure_id": "table_4_2", "chapter": "4.2"},
        "transition_ablation": {"figure_id": "table_4_4", "chapter": "4.4"},
    }

    for key, df in raw_data.items():
        if key in table_metadata:
            meta = table_metadata[key]
            df_copy = df.copy()
            df_copy["figure_id"] = meta["figure_id"]
            df_copy["chapter"] = meta["chapter"]
            df_copy["data_source"] = "summary_table"
            all_records.append(df_copy)
            print(f"  - 处理 {key}: {len(df)} 条记录")

    # 处理轨迹数据
    if not trajectory_df.empty:
        trajectory_df["data_source"] = "trajectory_data"
        trajectory_df["figure_id"] = "trajectory"
        all_records.append(trajectory_df)
        print(f"  - 处理轨迹数据: {len(trajectory_df)} 条点记录")

    if not all_records:
        print("  警告: 没有找到任何数据！")
        return pd.DataFrame()

    # 合并所有数据
    combined = pd.concat(all_records, ignore_index=True)
    combined["aggregated_at"] = datetime.now().isoformat()

    print(f"  总计: {len(combined)} 条记录")
    print()
    return combined


def save_aggregated_results(df: pd.DataFrame, trajectory_df: pd.DataFrame):
    """保存聚合结果"""
    print("[5/5] 保存聚合结果...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存主聚合文件
    df.to_csv(AGGREGATED_CSV, index=False, encoding="utf-8-sig")
    print(f"  - 主聚合文件: {AGGREGATED_CSV}")
    print(f"  - 文件大小: {AGGREGATED_CSV.stat().st_size / 1024:.1f} KB")

    # 保存轨迹数据单独文件
    if not trajectory_df.empty:
        trajectory_df.to_csv(TRAJECTORY_CSV, index=False, encoding="utf-8-sig")
        print(f"  - 轨迹数据文件: {TRAJECTORY_CSV}")
        print(f"  - 文件大小: {TRAJECTORY_CSV.stat().st_size / 1024:.1f} KB")

    print()

    # 生成数据摘要
    print("=" * 70)
    print("数据摘要")
    print("=" * 70)

    if "figure_id" in df.columns:
        print("\n数据来源分布:")
        fig_counts = df["figure_id"].value_counts()
        for fig_id, count in fig_counts.items():
            print(f"  - {fig_id}: {count}")

    if "scene" in df.columns:
        print("\n场景分布:")
        print(df["scene"].value_counts().to_string())

    if "method" in df.columns:
        print("\n方法分布:")
        print(df["method"].value_counts().head(10).to_string())

    if not trajectory_df.empty and "scenario" in trajectory_df.columns:
        print("\n轨迹数据:")
        traj_summary = trajectory_df.groupby("scenario")["trajectory_id"].nunique()
        print(traj_summary.to_string())

    print()
    print("=" * 70)
    print("聚合完成！")
    print("=" * 70)
    print()
    print(f"下一步: 运行 generate_paper_figures.py 生成论文图表")
    print()


def main():
    """主函数"""
    print("开始聚合实验结果...")
    print()

    # 检查源数据目录
    if not SOURCE_CSV_DIR.exists():
        print(f"错误: 找不到源数据目录: {SOURCE_CSV_DIR}")
        print("请确保 experiments/data_result/csv/ 目录存在")
        sys.exit(1)

    # 执行聚合流程
    available_scenarios = aggregate_from_scenarios()
    raw_data = load_raw_data()
    trajectory_df = load_trajectory_data(available_scenarios)
    aggregated_df = process_and_aggregate(raw_data, trajectory_df, available_scenarios)

    if aggregated_df.empty:
        print("错误: 未能成功聚合任何数据")
        sys.exit(1)

    save_aggregated_results(aggregated_df, trajectory_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
