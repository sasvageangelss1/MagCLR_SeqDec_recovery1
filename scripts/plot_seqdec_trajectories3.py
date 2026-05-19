"""
Thesis-ready trajectory plotting script — two-panel layout.

  (a) 场景1  — trajectory index 0 (first trajectory)
  (b) 场景2  — trajectory index 3 (fourth trajectory)

Each panel shows all four curves: Ground Truth, MagCLR + Regression,
MagCLR + Top-K Weighted Fusion, and SeqDec (Full).

Outputs
-------
scripts/paper_figures/
    fig_seqdec_trajectory_dual.pdf / .png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 统一论文风格
# ------------------------------------------------------------------
def _get_cn_font():
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return FontProperties(fname=fp)
    return None

CN_FONT = _get_cn_font()

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

# 字号与线宽
FS_LABEL  = 11
FS_TICK   = 9
FS_TITLE  = 10.5
FS_LEGEND = 8.5
SPINE_W   = 0.9
LINE_W    = 1.8
GT_W      = 1.4
TICK_LEN  = 3.5
PNG_DPI   = 600

# 配色
COLORS = {
    "gt":         "#000000",
    "regression": "#1f77b4",
    "retrieval":  "#ff7f0e",
    "seqdec":     "#2ca02c",
}

# 线型
LINESTYLES = {
    "gt":         "--",
    "regression": "-",
    "retrieval":  "-.",
    "seqdec":     "-",
}

ALPHA = {
    "gt":         0.85,
    "regression": 0.80,
    "retrieval":  0.80,
    "seqdec":     0.80,
}

METHOD_LABELS = {
    "regression": "SeqDec (w/o Jump Suppression)",
    "retrieval":  "SeqDec (w/o Displacement Consistency)",
    "seqdec":     "SeqDec (Full)",
    "gt":         "Ground Truth",
}

# ------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------
def load_trajectories():
    base = Path("experiments/scenario_1")

    reg_data = np.load(base / "eval_regression" / "test_regression_preds.npz")
    ret_data = np.load(base / "eval_retrieval"  / "test_retrieval_candidates.npz")
    sd_data  = np.load(base / "eval_seqdec"     / "test_seqdec_preds.npz")

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
    sd_seqs  = _split(sd_data["pred"])
    gt_seqs  = _split(reg_data["gt"])

    return dict(
        lengths=lengths,
        num_seqs=len(lengths),
        regression=reg_seqs,
        retrieval=ret_seqs,
        seqdec=sd_seqs,
        gt=gt_seqs,
    )


# ------------------------------------------------------------------
# 基础绘图工具
# ------------------------------------------------------------------
def _beautify(ax, xlabel=None, ylabel=None, xlim=None, ylim=None,
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


def _save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ------------------------------------------------------------------
# 坐标裁剪工具
# ------------------------------------------------------------------
def _clip_to_box(seq, xlim, ylim):
    """
    原来的 mask 裁剪函数保留，但不再用于画轨迹。

    原因：
    轨迹是连续线段，如果先用 mask 删除框外点，
    会导致穿过局部窗口的线段被切断，甚至完全不显示。

    局部放大图应当：
        先画完整轨迹；
        再通过 ax.set_xlim / ax.set_ylim 控制显示范围。
    """
    x0, x1 = xlim
    y0, y1 = ylim
    mask = (seq[:, 0] >= x0) & (seq[:, 0] <= x1) & \
           (seq[:, 1] >= y0) & (seq[:, 1] <= y1)
    return seq[mask]


# ------------------------------------------------------------------
# 双面板主函数
# ------------------------------------------------------------------
def plot_dual_trajectories(td):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)

    # (a) 场景1: 轨迹索引 0，第一张图，左上角区域 x∈(0,8), y∈(40,48) m
    # (b) 场景2: 轨迹索引 3，第四张图，右上角区域 x∈(32,40), y∈(40,48) m
    configs = [
        {"seq_i": 0, "label": "(a) 测试轨迹1", "xlim": (-0.5, 8),   "ylim": (40, 47)},
        {"seq_i": 3, "label": "(b) 测试轨迹2", "xlim": (32, 40),    "ylim": (40, 47)},
    ]

    # 两图的 y 显示范围都是 7 m，x 显示范围分别是 8.5 m 和 8 m
    # 为使两图物理高度完全一致，用 GridSpec 固定子图 data-box 宽高比
    # ratio = x_range / y_range；取较大者(8.5/7)作为统一基准
    y_range = 47 - 40  # 7 m
    x_ranges = [cfg["xlim"][1] - cfg["xlim"][0] for cfg in configs]  # [8.5, 8.0]

    for i, (ax, cfg) in enumerate(zip(axes, configs)):
        seq_i = cfg["seq_i"]
        xlim = cfg["xlim"]
        ylim = cfg["ylim"]

        # ----------------------------------------------------------
        # 关键修改：
        # 不再使用 _clip_to_box()。
        # 直接画完整轨迹，然后用 xlim / ylim 显示局部区域。
        # ----------------------------------------------------------
        gt_s  = td["gt"][seq_i]
        reg_s = td["regression"][seq_i]
        ret_s = td["retrieval"][seq_i]
        sd_s  = td["seqdec"][seq_i]

        ax.plot(gt_s[:, 0], gt_s[:, 1],
                color=COLORS["gt"], linestyle=LINESTYLES["gt"],
                linewidth=GT_W, label=METHOD_LABELS["gt"],
                alpha=ALPHA["gt"])

        for key, seq in [("regression", reg_s),
                         ("retrieval",  ret_s),
                         ("seqdec",     sd_s)]:
            ax.plot(seq[:, 0], seq[:, 1],
                    color=COLORS[key],
                    linestyle=LINESTYLES[key],
                    linewidth=LINE_W,
                    label=METHOD_LABELS[key],
                    alpha=ALPHA[key])

        _beautify(ax,
                  xlabel="x (m)",
                  ylabel="y (m)" if i == 0 else "",
                  xlim=xlim,
                  ylim=ylim,
                  equal_aspect=False)   # 改用 set_aspect 控制像素宽高比

        # -------------------------------------------------
        # 对齐核心：对每个子图单独设置 set_aspect，
        # 使数据区域的实际像素宽高比等于其 x_range / y_range
        # -------------------------------------------------
        ax.set_aspect((x_ranges[i] / y_range), adjustable="box")

        ax.set_title(cfg["label"], fontsize=FS_TITLE, fontproperties=CN_FONT, pad=6)

    # 统一图例：放在两个子图上方，不遮挡任何轨迹
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.52, 1.06),
               ncol=4, fontsize=FS_LEGEND,
               frameon=True, borderpad=0.3,
               handlelength=2.3, labelspacing=0.3, handletextpad=0.6)

    _save(fig, "fig_seqdec_trajectory_dual")


# ------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------
def main():
    print("Loading trajectory data …")
    td = load_trajectories()
    print(f"  {td['num_seqs']} trajectories, lengths={td['lengths']}")

    print("Plotting dual-panel trajectory comparison (Traj 1 & 4) …")
    plot_dual_trajectories(td)

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()