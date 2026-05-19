"""
论文图表生成脚本

从汇总的 aggregated_results.csv 读取数据，生成论文中所有图表和表格。
每个图和表格保存为独立的文件，命名遵循论文编号。

用法:
    python generate_paper_figures.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties

PROJECT_ROOT = Path(__file__).resolve().parent
AGGREGATED_CSV = PROJECT_ROOT / "thesis_paper_output" / "aggregated_results.csv"
SOURCE_CSV_DIR = PROJECT_ROOT / "experiments" / "data_result" / "csv"
OUTPUT_FIGURES = PROJECT_ROOT / "thesis_paper_output" / "figures"
OUTPUT_TABLES = PROJECT_ROOT / "thesis_paper_output" / "tables"

# 轨迹数据路径
SCENARIO_1_PATH = PROJECT_ROOT / "experiments" / "scenario_1"
SCENARIO_2_PATH = PROJECT_ROOT / "experiments" / "scenario_2"


# =============================================================================
# 统一论文风格配置
# =============================================================================
def get_cn_font():
    """获取中文字体"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return FontProperties(fname=fp)
    return None

CN_FONT = get_cn_font()

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 3
plt.rcParams["ps.fonttype"] = 3
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = False

if CN_FONT is not None:
    plt.rcParams["font.family"] = CN_FONT.get_name()
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]

FS_LABEL = 11
FS_TICK = 9
FS_TITLE = 10.5
FS_LEGEND = 8.5
SPINE_W = 0.9
LINE_W = 1.8
GT_W = 1.4
TICK_LEN = 3.5
PNG_DPI = 600

CDF_RANK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
CDF_RANK_LINESTYLES = ["-", "--", "-.", ":"]

METHOD_STYLE = {
    "RNN": {"ls": "-", "marker": None},
    "LSTM": {"ls": "--", "marker": None},
    "CNN+TCN": {"ls": "-.", "marker": None},
    "ConvNeXt-Lite-1D": {"ls": ":", "marker": None},
    "Scratch + MLP": {"ls": "--", "marker": None},
    "Pretrain + MLP": {"ls": "-", "marker": None},
    "In-Device Pretrain + FAISS": {"ls": "-", "marker": None},
    "LODO Pretrain + FAISS": {"ls": "--", "marker": None},
    "In-Device Pretrain + MLP": {"ls": "-.", "marker": None},
    "LODO Pretrain + MLP": {"ls": ":", "marker": None},
    "Full Model": {"ls": "-", "marker": None},
    "w/o Equal-Distance": {"ls": "--", "marker": None},
    "w/o Augmentation": {"ls": "--", "marker": None},
    "w/o Local-Variation Features": {"ls": ":", "marker": None},
    "WKNN (Fingerprint)": {"ls": "--", "marker": None, "color": "#999999"},
    "PDR (Step-Heading)": {"ls": ":", "marker": None, "color": "#aaaaaa"},
    "MagCLR + Regression": {"ls": "-.", "marker": None},
    "MagCLR + Top-K Weighted Fusion": {"ls": "--", "marker": None},
    "SeqDec": {"ls": "-", "marker": None},
    "SeqDec (Full)": {"ls": "-", "marker": None},
    "SeqDec (w/o Confidence)": {"ls": "--", "marker": None},
    "SeqDec (w/o Displacement Consistency)": {"ls": "--", "marker": None},
    "SeqDec (w/o Jump Suppression)": {"ls": ":", "marker": None},
    "Ground Truth": {"ls": "--", "marker": None},
}


# =============================================================================
# 工具函数
# =============================================================================
def read_csv_safe(path: Path) -> pd.DataFrame:
    """安全读取 CSV"""
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(f"无法读取文件: {path}")


def get_style(label: str):
    return METHOD_STYLE.get(label, {"ls": "-", "marker": None})


def beautify_axes(ax, xlabel=None, ylabel=None, xlim=None, ylim=None,
                  equal_aspect=False, add_ygrid=False):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL, fontproperties=CN_FONT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontproperties=CN_FONT)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="both", which="major", labelsize=FS_TICK,
                   direction="in", length=TICK_LEN, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.0, width=0.7)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(SPINE_W)

    if add_ygrid:
        ax.yaxis.grid(True, linestyle=(0, (2, 2)), linewidth=0.5, alpha=0.35)
    else:
        ax.grid(False)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def style_legend(ax, loc="best", ncol=1, fontsize=FS_LEGEND):
    leg = ax.legend(loc=loc, fontsize=fontsize, ncol=ncol,
                    frameon=True, borderpad=0.3, handlelength=2.3,
                    labelspacing=0.3, handletextpad=0.6)
    if leg is not None:
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_edgecolor("black")
    return leg


def save_figure(fig, stem: str):
    """保存图表到独立文件夹"""
    output_dir = OUTPUT_FIGURES / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_table(df: pd.DataFrame, stem: str, index=False):
    """保存表格到独立文件夹"""
    output_dir = OUTPUT_TABLES / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=index, encoding="utf-8-sig")


def plot_ecdf(ax, data, label, color=None, linestyle=None):
    arr = pd.Series(data).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return

    arr = np.sort(arr)
    y = np.arange(1, len(arr) + 1) / len(arr)
    style = get_style(label)

    kwargs = {
        "label": label,
        "linewidth": LINE_W,
        "linestyle": linestyle if linestyle is not None else style["ls"],
    }
    if color:
        kwargs["color"] = color

    ax.step(arr, y, where="post", **kwargs)


# =============================================================================
# 图表生成函数
# =============================================================================
def generate_figure_3_7_2_backbone_cdf(raw_df: pd.DataFrame):
    """图 3.7.2: 骨干网络性能对比 CDF"""
    print("  生成图 3.7.2 (骨干网络性能对比)...")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)

    sub_df = raw_df[raw_df["figure_id"] == "3.7.2_E1_backbone"]
    methods = ["ConvNeXt-Lite-1D", "CNN+TCN", "LSTM", "RNN"]

    for i, (ax, scene, xmax) in enumerate([
        (axes[0], "场景1", 7.0),
        (axes[1], "场景2", 8.0)
    ]):
        scene_sub = sub_df[sub_df["scene"] == scene]
        for color, ls, method in zip(CDF_RANK_COLORS, CDF_RANK_LINESTYLES, methods):
            data = scene_sub[scene_sub["method"] == method]["error_m"]
            plot_ecdf(ax, data, method, color=color, linestyle=ls)

        beautify_axes(ax, xlabel="定位误差/m", ylabel="概率" if i == 0 else "",
                      xlim=(0, xmax), ylim=(0, 1.01), add_ygrid=True)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_title(f"({chr(97+i)}) {scene}", fontsize=FS_TITLE, fontproperties=CN_FONT)

    style_legend(axes[0], loc="lower right", fontsize=8.0)
    save_figure(fig, "fig_3_7_2_backbone_cdf")
    save_table(pd.DataFrame(), "fig_3_7_2_backbone_cdf")


def generate_figure_3_7_3_pretrain_cdf(raw_df: pd.DataFrame):
    """图 3.7.3: 预训练方法对比 CDF"""
    print("  生成图 3.7.3 (预训练方法对比)...")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), constrained_layout=True)

    sub_df = raw_df[raw_df["figure_id"] == "3.7.3_E2_pretrain"]
    methods = ["Pretrain + MLP", "Scratch + MLP"]

    for i, (ax, scene, xmax) in enumerate([
        (axes[0], "场景1", 3.2),
        (axes[1], "场景2", 4.6)
    ]):
        scene_sub = sub_df[sub_df["scene"] == scene]
        for color, method in zip(CDF_RANK_COLORS[:2], methods):
            data = scene_sub[scene_sub["method"] == method]["error_m"]
            plot_ecdf(ax, data, method, color=color)

        beautify_axes(ax, xlabel="定位误差/m", ylabel="概率" if i == 0 else "",
                      xlim=(0, xmax), ylim=(0, 1.01), add_ygrid=True)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_title(f"({chr(97+i)}) {scene}", fontsize=FS_TITLE, fontproperties=CN_FONT)

    style_legend(axes[0], loc="lower right")
    save_figure(fig, "fig_3_7_3_pretrain_cdf")


def generate_figure_3_7_5_downstream_cdf(raw_df: pd.DataFrame):
    """图 3.7.5: 跨设备定位性能 CDF"""
    print("  生成图 3.7.5 (跨设备定位性能)...")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)

    sub_df = raw_df[raw_df["figure_id"] == "3.7.5_downstream_cross_device"]
    methods = [
        "In-Device Pretrain + FAISS",
        "LODO Pretrain + FAISS",
        "In-Device Pretrain + MLP",
        "LODO Pretrain + MLP"
    ]

    for i, (ax, scene, xmax) in enumerate([
        (axes[0], "场景1", 8.2),
        (axes[1], "场景2", 8.9)
    ]):
        scene_sub = sub_df[sub_df["scene"] == scene]
        for color, ls, method in zip(CDF_RANK_COLORS, CDF_RANK_LINESTYLES, methods):
            data = scene_sub[scene_sub["method"] == method]["error_m"]
            plot_ecdf(ax, data, method, color=color, linestyle=ls)

        beautify_axes(ax, xlabel="定位误差/m", ylabel="概率" if i == 0 else "",
                      xlim=(0, xmax), ylim=(0, 1.01), add_ygrid=True)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_title(f"({chr(97+i)}) {scene}", fontsize=FS_TITLE, fontproperties=CN_FONT)

    style_legend(axes[0], loc="lower right", fontsize=7.6)
    save_figure(fig, "fig_3_7_5_downstream_cross_device_cdf")


def generate_figure_3_7_6_A1_cdf(raw_df: pd.DataFrame):
    """图 3.7.6 A1: 等间距采样消融 CDF"""
    print("  生成图 3.7.6 A1 (等间距采样消融)...")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), constrained_layout=True)

    sub_df = raw_df[raw_df["figure_id"] == "3.7.6_A1_equal_distance"]
    methods = ["Full Model", "w/o Equal-Distance"]

    for i, (ax, scene, xmax) in enumerate([
        (axes[0], "场景1", 5.7),
        (axes[1], "场景2", 6.7)
    ]):
        scene_sub = sub_df[sub_df["scene"] == scene]
        for color, method in zip(CDF_RANK_COLORS[:2], methods):
            data = scene_sub[scene_sub["method"] == method]["error_m"]
            plot_ecdf(ax, data, method, color=color)

        beautify_axes(ax, xlabel="定位误差/m", ylabel="概率" if i == 0 else "",
                      xlim=(0, xmax), ylim=(0, 1.01), add_ygrid=True)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_title(f"({chr(97+i)}) {scene}", fontsize=FS_TITLE, fontproperties=CN_FONT)

    style_legend(axes[0], loc="lower right")
    save_figure(fig, "fig_3_7_6_A1_equal_distance_cdf")


def generate_figure_3_7_6_A2_cdf(raw_df: pd.DataFrame):
    """图 3.7.6 A2: 表征消融 CDF"""
    print("  生成图 3.7.6 A2 (表征消融)...")

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)

    sub_df = raw_df[raw_df["figure_id"] == "3.7.6_A2_representation_ablation"]
    methods = ["Full Model", "w/o Augmentation", "w/o Local-Variation Features"]

    for color, ls, method in zip(CDF_RANK_COLORS, CDF_RANK_LINESTYLES, methods):
        data = sub_df[sub_df["method"] == method]["error_m"]
        plot_ecdf(ax, data, method, color=color, linestyle=ls)

    beautify_axes(ax, xlabel="定位误差/m", ylabel="概率", xlim=(0, 3.1), ylim=(0, 1.01), add_ygrid=True)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_title("场景1", fontsize=FS_TITLE, fontproperties=CN_FONT)
    style_legend(ax, loc="lower right")
    save_figure(fig, "fig_3_7_6_A2_representation_ablation_cdf")


def generate_figure_4_7_3_continuous_cdf(raw_df: pd.DataFrame):
    """图 4.7.3: SeqDec 连续定位性能 CDF（含 WKNN 和 PDR 基线）"""
    print("  生成图 4.7.3 (SeqDec 连续定位性能，含基线对比)...")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    sub_df = raw_df[raw_df["figure_id"] == "4.7.3_continuous"]

    # Method order: baseline first (WKNN, PDR), then proposed methods
    methods = [
        "WKNN (Fingerprint)",
        "PDR (Step-Heading)",
        "MagCLR + Regression",
        "MagCLR + Top-K Weighted Fusion",
        "SeqDec",
    ]
    # Colors: gray for baselines, blue→green gradient for MagCLR methods
    colors = [
        "#999999",   # WKNN — grey
        "#aaaaaa",   # PDR  — light grey
        "#1f77b4",   # Regression — blue
        "#ff7f0e",   # Weighted Fusion — orange
        "#2ca02c",   # SeqDec — green (best)
    ]
    linestyles = ["--", ":", "-.", "--", "-"]

    for i, (ax, scene, xmax) in enumerate([
        (axes[0], "场景1", 8.5),
        (axes[1], "场景2", 8.5)
    ]):
        scene_sub = sub_df[sub_df["scene"] == scene]
        for color, ls, method in zip(colors, linestyles, methods):
            data = scene_sub[scene_sub["method"] == method]["error_m"]
            if len(data) == 0:
                print(f"    [warn] no data for method={method!r}, scene={scene!r}")
                continue
            plot_ecdf(ax, data, method, color=color, linestyle=ls)

        beautify_axes(ax, xlabel="连续定位误差/m", ylabel="概率" if i == 0 else "",
                      xlim=(0, xmax), ylim=(0, 1.01), add_ygrid=True)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_title(f"({chr(97+i)}) {scene}", fontsize=FS_TITLE, fontproperties=CN_FONT)

    style_legend(axes[0], loc="lower right", fontsize=7.5)
    save_figure(fig, "fig_4_7_3_continuous_cdf")


def generate_figure_4_7_4_confidence_boxplot(boxplot_df: pd.DataFrame):
    """图 4.7.4: 置信度分组箱线图"""
    print("  生成图 4.7.4 (置信度分析箱线图)...")

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)

    order_groups = ["高置信度", "中置信度", "低置信度"]
    method_colors = {
        "SeqDec (Full)": "#2ca02c",
        "SeqDec (w/o Confidence)": "#ff7f0e",
    }

    positions, data, labels, box_colors = [], [], [], []
    pos = 1.0
    for group in order_groups:
        for method in ["SeqDec (Full)", "SeqDec (w/o Confidence)"]:
            arr = boxplot_df[(boxplot_df["confidence_group"] == group) &
                            (boxplot_df["method"] == method)]["error_m"].values
            data.append(arr)
            positions.append(pos)
            labels.append(f"{group}\n{'Full' if method == 'SeqDec (Full)' else 'w/o Conf.'}")
            box_colors.append(method_colors[method])
            pos += 1.0
        pos += 0.45

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.2, "color": "black"},
        whiskerprops={"linewidth": 0.9, "color": "black"},
        capprops={"linewidth": 0.9, "color": "black"},
        boxprops={"linewidth": 0.9, "color": "black"},
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    beautify_axes(ax, ylabel="定位误差 (m)", ylim=(0, 2.4), add_ygrid=True)
    save_figure(fig, "fig_4_7_4_confidence_boxplot")


def generate_figure_4_7_6_lambda_dual_axis(lambda_df: pd.DataFrame):
    """图 4.7.6: Lambda 参数敏感性双轴图"""
    print("  生成图 4.7.6 (Lambda 敏感性分析)...")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)

    mae_color = "#1f77b4"
    jump_color = "#ff7f0e"

    for i, scene in enumerate(["场景1", "场景2"]):
        sub = lambda_df[lambda_df["场景"] == scene].sort_values("lambda")

        ax = axes[i]
        ax2 = ax.twinx()

        l1 = ax.plot(
            sub["lambda"], sub["MAE (m)"],
            color=mae_color, linestyle="-", marker="o", markersize=4.8,
            markerfacecolor="white", markeredgewidth=1.0, linewidth=1.8,
            label="MAE"
        )

        l2 = ax2.plot(
            sub["lambda"], sub["跳变比例 (%)"],
            color=jump_color, linestyle="--", marker="s", markersize=4.6,
            markerfacecolor="white", markeredgewidth=1.0, linewidth=1.6,
            label="跳变比例"
        )

        ax.set_xlabel(r"$\lambda$", fontsize=FS_LABEL)
        ax.set_xticks(sub["lambda"])

        if i == 0:
            ax.set_ylabel("平均定位误差/m", fontsize=FS_LABEL, fontproperties=CN_FONT)
            ax2.set_ylabel("")
        else:
            ax.set_ylabel("")
            ax2.set_ylabel("跳变比例/%", fontsize=FS_LABEL, fontproperties=CN_FONT)

        ax.set_title(f"({chr(97+i)}) {scene}", fontsize=FS_TITLE, fontproperties=CN_FONT)

        ax.tick_params(axis="both", which="major", labelsize=FS_TICK, direction="in", length=TICK_LEN, width=0.8)
        ax.tick_params(axis="both", which="minor", direction="in", length=2.0, width=0.7)
        ax2.tick_params(axis="y", which="major", labelsize=FS_TICK, direction="in", length=TICK_LEN, width=0.8)

        ax.yaxis.grid(True, linestyle=(0, (2, 2)), linewidth=0.5, alpha=0.35)
        ax.grid(False)
        ax2.grid(False)

        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(SPINE_W)
        for sp in ax2.spines.values():
            sp.set_linewidth(SPINE_W)

        if i == 0:
            lines = l1 + l2
            labels_leg = [line.get_label() for line in lines]
            leg = ax.legend(lines, labels_leg, loc="upper right", fontsize=FS_LEGEND,
                           frameon=True, borderpad=0.3, handlelength=2.3,
                           labelspacing=0.3, handletextpad=0.6, prop=CN_FONT)
            leg.get_frame().set_linewidth(0.7)
            leg.get_frame().set_alpha(1.0)
            leg.get_frame().set_edgecolor("black")

    save_figure(fig, "fig_4_7_6_lambda_dual_axis")


def generate_table_3_4_backbone():
    """表 3.4: 骨干网络性能对比表"""
    print("  生成表 3.4 (骨干网络性能对比)...")

    # 从原始数据读取
    cdf_file = SOURCE_CSV_DIR / "cdf_raw_all_v7_replaced.csv"
    if cdf_file.exists():
        df = read_csv_safe(cdf_file)
        backbone_df = df[df["figure_id"] == "3.7.2_E1_backbone"]

        records = []
        for scene in ["场景1", "场景2"]:
            for method in ["RNN", "LSTM", "CNN+TCN", "ConvNeXt-Lite-1D"]:
                data = backbone_df[(backbone_df["scene"] == scene) &
                                  (backbone_df["method"] == method)]["error_m"]
                if len(data) > 0:
                    records.append({
                        "网络类型": method,
                        "场景": scene,
                        "平均误差": f"{data.mean():.2f}",
                        "中位误差": f"{data.median():.2f}",
                        "P90误差": f"{np.percentile(data, 90):.2f}",
                    })

        result_df = pd.DataFrame(records)
        save_table(result_df, "table_3_4_backbone_perf")


def generate_table_4_2_continuous():
    """表 4.2: 连续定位性能对比表（含 WKNN 和 PDR 基线）"""
    print("  生成表 4.2 (连续定位性能对比)...")

    cdf_file = SOURCE_CSV_DIR / "cdf_raw_all_v7_replaced.csv"
    if cdf_file.exists():
        df = read_csv_safe(cdf_file)
        seq_df = df[df["figure_id"] == "4.7.3_continuous"]

        records = []
        for scene in ["场景1", "场景2"]:
            for method in [
                "WKNN (Fingerprint)",
                "PDR (Step-Heading)",
                "MagCLR + Regression",
                "MagCLR + Top-K Weighted Fusion",
                "SeqDec",
            ]:
                data = seq_df[(seq_df["scene"] == scene) &
                             (seq_df["method"] == method)]["error_m"]
                if len(data) > 0:
                    records.append({
                        "场景": scene,
                        "方法": method,
                        "平均误差/m": f"{data.mean():.3f}",
                        "中位误差/m": f"{data.median():.3f}",
                        "P90误差/m": f"{np.percentile(data, 90):.3f}",
                    })

        result_df = pd.DataFrame(records)
        save_table(result_df, "table_4_2_continuous_perf")


def generate_table_4_3_confidence():
    """表 4.3: 置信度分析对比表"""
    print("  生成表 4.3 (置信度分析)...")

    box_file = SOURCE_CSV_DIR / "boxplot_raw_confidence_v5.csv"
    if box_file.exists():
        df = read_csv_safe(box_file)

        records = []
        for group in ["高置信度", "中置信度", "低置信度"]:
            for method in ["SeqDec (Full)", "SeqDec (w/o Confidence)"]:
                data = df[(df["confidence_group"] == group) &
                         (df["method"] == method)]["error_m"]
                if len(data) > 0:
                    records.append({
                        "置信度分组": group,
                        "方法": method,
                        "平均误差": f"{data.mean():.3f}",
                        "中位误差": f"{data.median():.3f}",
                        "样本数": len(data),
                    })

        result_df = pd.DataFrame(records)
        save_table(result_df, "table_4_3_confidence_analysis")


def generate_table_4_4_transition_ablation():
    """表 4.4: 状态转移模块消融实验表"""
    print("  生成表 4.4 (状态转移模块消融)...")

    table_file = SOURCE_CSV_DIR / "table_4_7_5_transition_ablation_v5.csv"
    if table_file.exists():
        df = read_csv_safe(table_file)
        save_table(df, "table_4_4_transition_ablation")


# =============================================================================
# 轨迹图生成函数
# =============================================================================

def load_trajectory_data(scenario_path: Path) -> dict:
    """加载轨迹数据"""
    reg_path = scenario_path / "eval_regression" / "test_regression_preds.npz"
    ret_path = scenario_path / "eval_retrieval" / "test_retrieval_candidates.npz"
    sd_path = scenario_path / "eval_seqdec" / "test_seqdec_preds.npz"

    if not all(p.exists() for p in [reg_path, ret_path, sd_path]):
        return None

    reg_data = np.load(reg_path)
    ret_data = np.load(ret_path)
    sd_data = np.load(sd_path)

    lengths = ret_data["lengths"].tolist()

    def _split(arr):
        arr = np.asarray(arr)
        result, pos = [], 0
        for ln in lengths:
            result.append(arr[pos:pos + ln].copy())
            pos += ln
        return result

    return dict(
        lengths=lengths,
        num_seqs=len(lengths),
        regression=_split(reg_data["pred"]),
        retrieval=_split(ret_data["pred"]),
        seqdec=_split(sd_data["pred"]),
        gt=_split(reg_data["gt"]),
    )


def generate_figure_4_9_trajectory_scene1(td: dict):
    """图 4.9: 低置信度磁场相似片段下的轨迹修正示例 (场景1)"""
    print("  生成图 4.9 (场景1轨迹图)...")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=False)

    traj_indices = [1, 2, 3]
    labels = ["(a) 场景1-测试轨迹1", "(b) 场景1-测试轨迹2", "(c) 场景1-测试轨迹3"]

    for ax_i, (ax, seq_i, label) in enumerate(zip(axes, traj_indices, labels)):
        gt_s = td["gt"][seq_i]
        reg_s = td["regression"][seq_i]
        ret_s = td["retrieval"][seq_i]
        sd_s = td["seqdec"][seq_i]

        ax.plot(gt_s[:, 0], gt_s[:, 1],
                color="#000000", linestyle="--", linewidth=1.4,
                label="Ground Truth", alpha=0.85)

        ax.plot(reg_s[:, 0], reg_s[:, 1], color="#1f77b4", linestyle="-",
                linewidth=1.8, label="MagCLR + Regression", alpha=0.80)
        ax.plot(ret_s[:, 0], ret_s[:, 1], color="#ff7f0e", linestyle="-.",
                linewidth=1.8, label="MagCLR + Top-K Weighted Fusion", alpha=0.80)
        ax.plot(sd_s[:, 0], sd_s[:, 1], color="#2ca02c", linestyle="-",
                linewidth=1.8, label="SeqDec (Full)", alpha=0.80)

        all_x = np.concatenate([gt_s[:, 0], reg_s[:, 0], ret_s[:, 0], sd_s[:, 0]])
        all_y = np.concatenate([gt_s[:, 1], reg_s[:, 1], ret_s[:, 1], sd_s[:, 1]])
        x_pad = (all_x.max() - all_x.min()) * 0.08
        y_pad = (all_y.max() - all_y.min()) * 0.08

        ax.set_xlabel("x (m)", fontsize=11, fontproperties=CN_FONT)
        if ax_i == 0:
            ax.set_ylabel("y (m)", fontsize=11, fontproperties=CN_FONT)
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
        ax.set_aspect("equal", adjustable="box")

        ax.tick_params(axis="both", which="major", labelsize=9, direction="in", length=3.5, width=0.8)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
        ax.grid(False)
        ax.set_title(label, fontsize=10.5, fontproperties=CN_FONT)

        if ax_i == 0:
            leg = ax.legend(loc="center", bbox_to_anchor=(0.55, 0.42),
                          fontsize=8.0, frameon=True, borderpad=0.3,
                          handlelength=2.3, labelspacing=0.3, handletextpad=0.6)
            if leg:
                leg.get_frame().set_linewidth(0.7)
                leg.get_frame().set_alpha(1.0)
                leg.get_frame().set_edgecolor("black")

    fig.subplots_adjust(wspace=0.04)
    save_figure(fig, "fig_4_9_low_confidence_trajectory_scene1")


def generate_figure_4_9_trajectory_scene2(td: dict):
    """图 4.9 续: 场景2轨迹图"""
    print("  生成图 4.9 (场景2轨迹图)...")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=False)

    traj_indices = [0, 1, 5]
    labels = ["(a) 场景2-测试轨迹1", "(b) 场景2-测试轨迹2", "(c) 场景2-测试轨迹3"]

    for ax_i, (ax, seq_i, label) in enumerate(zip(axes, traj_indices, labels)):
        gt_s = td["gt"][seq_i]
        reg_s = td["regression"][seq_i]
        ret_s = td["retrieval"][seq_i]
        sd_s = td["seqdec"][seq_i]

        ax.plot(gt_s[:, 0], gt_s[:, 1],
                color="#000000", linestyle="--", linewidth=1.4,
                label="Ground Truth", alpha=0.85)

        ax.plot(reg_s[:, 0], reg_s[:, 1], color="#1f77b4", linestyle="-",
                linewidth=1.8, label="MagCLR + Regression", alpha=0.80)
        ax.plot(ret_s[:, 0], ret_s[:, 1], color="#ff7f0e", linestyle="-.",
                linewidth=1.8, label="MagCLR + Top-K Weighted Fusion", alpha=0.80)
        ax.plot(sd_s[:, 0], sd_s[:, 1], color="#2ca02c", linestyle="-",
                linewidth=1.8, label="SeqDec (Full)", alpha=0.80)

        all_x = np.concatenate([gt_s[:, 0], reg_s[:, 0], ret_s[:, 0], sd_s[:, 0]])
        all_y = np.concatenate([gt_s[:, 1], reg_s[:, 1], ret_s[:, 1], sd_s[:, 1]])
        x_pad = (all_x.max() - all_x.min()) * 0.08
        y_pad = (all_y.max() - all_y.min()) * 0.08

        ax.set_xlabel("x (m)", fontsize=11, fontproperties=CN_FONT)
        if ax_i == 0:
            ax.set_ylabel("y (m)", fontsize=11, fontproperties=CN_FONT)
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
        ax.set_aspect("equal", adjustable="box")

        ax.tick_params(axis="both", which="major", labelsize=9, direction="in", length=3.5, width=0.8)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
        ax.grid(False)
        ax.set_title(label, fontsize=10.5, fontproperties=CN_FONT)

        if ax_i == 0:
            leg = ax.legend(loc="center", bbox_to_anchor=(0.55, 0.42),
                          fontsize=8.0, frameon=True, borderpad=0.3,
                          handlelength=2.3, labelspacing=0.3, handletextpad=0.6)
            if leg:
                leg.get_frame().set_linewidth(0.7)
                leg.get_frame().set_alpha(1.0)
                leg.get_frame().set_edgecolor("black")

    fig.subplots_adjust(wspace=0.04)
    save_figure(fig, "fig_4_9_low_confidence_trajectory_scene2")


def generate_figure_4_11_jump_analysis(td: dict):
    """图 4.11: 跳变高发风险片段与失败案例分析"""
    print("  生成图 4.11 (跳变高发风险片段分析)...")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)

    configs = [
        {"seq_i": 0, "label": "(a) 测试轨迹1", "xlim": (-0.5, 8), "ylim": (40, 47)},
        {"seq_i": 3, "label": "(b) 测试轨迹2", "xlim": (32, 40), "ylim": (40, 47)},
    ]

    y_range = 47 - 40
    x_ranges = [cfg["xlim"][1] - cfg["xlim"][0] for cfg in configs]

    for i, (ax, cfg) in enumerate(zip(axes, configs)):
        seq_i = cfg["seq_i"]
        xlim = cfg["xlim"]
        ylim = cfg["ylim"]

        gt_s = td["gt"][seq_i]
        reg_s = td["regression"][seq_i]
        ret_s = td["retrieval"][seq_i]
        sd_s = td["seqdec"][seq_i]

        ax.plot(gt_s[:, 0], gt_s[:, 1], color="#000000", linestyle="--",
                linewidth=1.4, label="Ground Truth", alpha=0.85)

        ax.plot(reg_s[:, 0], reg_s[:, 1], color="#1f77b4", linestyle="-",
                linewidth=1.8, label="SeqDec (w/o Jump Suppression)", alpha=0.80)
        ax.plot(ret_s[:, 0], ret_s[:, 1], color="#ff7f0e", linestyle="-.",
                linewidth=1.8, label="SeqDec (w/o Displacement Consistency)", alpha=0.80)
        ax.plot(sd_s[:, 0], sd_s[:, 1], color="#2ca02c", linestyle="-",
                linewidth=1.8, label="SeqDec (Full)", alpha=0.80)

        ax.set_xlabel("x (m)", fontsize=11, fontproperties=CN_FONT)
        if i == 0:
            ax.set_ylabel("y (m)", fontsize=11, fontproperties=CN_FONT)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect((x_ranges[i] / y_range), adjustable="box")

        ax.tick_params(axis="both", which="major", labelsize=9, direction="in", length=3.5, width=0.8)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
        ax.grid(False)
        ax.set_title(cfg["label"], fontsize=10.5, fontproperties=CN_FONT, pad=6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.06),
               ncol=4, fontsize=8.5, frameon=True, borderpad=0.3,
               handlelength=2.3, labelspacing=0.3, handletextpad=0.6)

    save_figure(fig, "fig_4_11_jump_risk_analysis")


def generate_table_4_5_lambda_sensitivity():
    """表 4.5: Lambda 敏感性分析表"""
    print("  生成表 4.5 (Lambda 敏感性)...")

    lambda_file = SOURCE_CSV_DIR / "lambda_sensitivity_v8.csv"
    if lambda_file.exists():
        df = read_csv_safe(lambda_file)
        save_table(df, "table_4_5_lambda_sensitivity")


# =============================================================================
# 主函数
# =============================================================================
def main():
    """主函数"""
    print("[1/4] 加载聚合数据...")
    print()

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    raw_df = read_csv_safe(SOURCE_CSV_DIR / "cdf_raw_all_v7_replaced.csv")
    boxplot_df = read_csv_safe(SOURCE_CSV_DIR / "boxplot_raw_confidence_v5.csv")
    lambda_df = read_csv_safe(SOURCE_CSV_DIR / "lambda_sensitivity_v8.csv")

    print(f"  - CDF 数据: {len(raw_df)} 条记录")
    print(f"  - 置信度数据: {len(boxplot_df)} 条记录")
    print(f"  - Lambda 数据: {len(lambda_df)} 条记录")

    # 加载轨迹数据
    print()
    print("[2/4] 加载轨迹数据...")
    print()

    td_scene1 = load_trajectory_data(SCENARIO_1_PATH)
    td_scene2 = load_trajectory_data(SCENARIO_2_PATH)

    if td_scene1:
        print(f"  - 场景1: {td_scene1['num_seqs']} 条轨迹")
    if td_scene2:
        print(f"  - 场景2: {td_scene2['num_seqs']} 条轨迹")

    print()
    print("[3/4] 生成论文图表...")
    print()

    generate_figure_3_7_2_backbone_cdf(raw_df)
    generate_figure_3_7_3_pretrain_cdf(raw_df)
    generate_figure_3_7_5_downstream_cdf(raw_df)
    generate_figure_3_7_6_A1_cdf(raw_df)
    generate_figure_3_7_6_A2_cdf(raw_df)
    generate_figure_4_7_3_continuous_cdf(raw_df)
    generate_figure_4_7_4_confidence_boxplot(boxplot_df)
    generate_figure_4_7_6_lambda_dual_axis(lambda_df)

    # 生成轨迹图
    print()
    print("[3.5/4] 生成轨迹图...")
    print()

    if td_scene1:
        generate_figure_4_9_trajectory_scene1(td_scene1)
    if td_scene2:
        generate_figure_4_9_trajectory_scene2(td_scene2)
    if td_scene1:
        generate_figure_4_11_jump_analysis(td_scene1)

    print()
    print("[4/4] 生成论文表格...")
    print()

    generate_table_3_4_backbone()
    generate_table_4_2_continuous()
    generate_table_4_3_confidence()
    generate_table_4_4_transition_ablation()
    generate_table_4_5_lambda_sensitivity()

    print()
    print("=" * 70)
    print("生成完成！")
    print("=" * 70)
    print()
    print(f"图表目录: {OUTPUT_FIGURES}")
    print(f"表格目录: {OUTPUT_TABLES}")
    print()

    print("生成的文件夹结构:")
    print("-" * 50)

    # 列出图表文件夹
    fig_folders = sorted([d for d in OUTPUT_FIGURES.iterdir() if d.is_dir()])
    if fig_folders:
        print(f"\n图表文件夹 ({len(fig_folders)} 个):")
        for d in fig_folders:
            files = list(d.glob("*"))
            print(f"  [{d.name}]/")
            for f in sorted(files):
                print(f"      - {f.name}")

    # 列出表格文件夹
    tbl_folders = sorted([d for d in OUTPUT_TABLES.iterdir() if d.is_dir()])
    if tbl_folders:
        print(f"\n表格文件夹 ({len(tbl_folders)} 个):")
        for d in tbl_folders:
            files = list(d.glob("*"))
            print(f"  [{d.name}]/")
            for f in sorted(files):
                print(f"      - {f.name}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
