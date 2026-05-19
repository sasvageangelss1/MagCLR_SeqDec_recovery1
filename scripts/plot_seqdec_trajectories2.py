"""
Thesis-ready trajectory plotting script — scenario_1 test split.

Mimics the figure style of the reference CDF/trajectory plotting script:
  • Unified thesis font / spines / tick / line-width settings
  • Chinese font support
  • Exports PDF (vector) + PNG (600 DPI) simultaneously
  • Three-panel triangular layout: (a) Traj 1  (b) Traj 2  (c) Traj 3

Outputs
-------
scripts/paper_figures/
    fig_seqdec_trajectory_tri.pdf / .png
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
# 统一论文风格（与 run_paper_experiments.py 保持一致）
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
plt.rcParams["pdf.fonttype"] = 3      # Type-3 avoids TTC embedding issues; vector quality sufficient
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

# 字号与线宽（与参考脚本一致）
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

# 线型（GT=虚线，三种方法=不同线型）
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
    "regression": "MagCLR + Regression",
    "retrieval":  "MagCLR + Top-K Weighted Fusion",
    "seqdec":     "SeqDec (Full)",
    "gt":         "Ground Truth",
}

# ------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------
def load_trajectories():
    base = Path("experiments/scenario_2")

    reg_data = np.load(base / "eval_regression" / "test_regression_preds.npz")
    ret_data = np.load(base / "eval_retrieval"  / "test_retrieval_candidates.npz")
    sd_data  = np.load(base / "eval_seqdec"     / "test_seqdec_preds.npz")

    lengths = ret_data["lengths"].tolist()
    num_seqs = len(lengths)

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
        num_seqs=num_seqs,
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


def _style_legend(ax, loc="best", ncol=1, fontsize=FS_LEGEND):
    leg = ax.legend(loc=loc, fontsize=fontsize, ncol=ncol,
                   frameon=True, borderpad=0.3, handlelength=2.3,
                   labelspacing=0.3, handletextpad=0.6)
    if leg is not None:
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_edgecolor("black")
    return leg


def _save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ------------------------------------------------------------------
# 辅助：画单条轨迹的误差标注
# ------------------------------------------------------------------
def _annotate_errors(ax, gt_seq, reg_seq, ret_seq, sd_seq):
    err_reg = np.linalg.norm(reg_seq - gt_seq, axis=1).mean()
    err_ret = np.linalg.norm(ret_seq  - gt_seq, axis=1).mean()
    err_sd  = np.linalg.norm(sd_seq   - gt_seq, axis=1).mean()
    return err_reg, err_ret, err_sd


# ------------------------------------------------------------------
def plot_tri_trajectories(td):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=False)

    # Traj 2 = index 1, Traj 3 = index 2, Traj 4 = index 3
    traj_indices = [0, 1, 5]
    # traj_indices = [1, 2, 3]
    labels = ["(a) 场景2-测试轨迹1", "(b) 场景2-测试轨迹2", "(c) 场景2-测试轨迹3"]

    for ax_i, (ax, seq_i, label) in enumerate(zip(axes, traj_indices, labels)):
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

        # 统一坐标范围 + 8% 边距
        all_x = np.concatenate([gt_s[:, 0], reg_s[:, 0], ret_s[:, 0], sd_s[:, 0]])
        all_y = np.concatenate([gt_s[:, 1], reg_s[:, 1], ret_s[:, 1], sd_s[:, 1]])
        x_pad = (all_x.max() - all_x.min()) * 0.08
        y_pad = (all_y.max() - all_y.min()) * 0.08

        _beautify(ax,
                  xlabel="x (m)",
                  ylabel="y (m)" if ax_i == 0 else "",
                  xlim=(all_x.min() - x_pad, all_x.max() + x_pad),
                  ylim=(all_y.min() - y_pad, all_y.max() + y_pad),
                  equal_aspect=True)

        ax.set_title(label, fontsize=FS_TITLE, fontproperties=CN_FONT)

        # 图例只放第一个子图
        if ax_i == 0:
            leg = ax.legend(
                loc="center", bbox_to_anchor=(0.55, 0.42),
                fontsize=8.0, frameon=True, borderpad=0.3,
                handlelength=2.3, labelspacing=0.3, handletextpad=0.6,
            )
            if leg:
                leg.get_frame().set_linewidth(0.7)
                leg.get_frame().set_alpha(1.0)
                leg.get_frame().set_edgecolor("black")

    fig.subplots_adjust(wspace=0.04)
    _save(fig, "fig_seqdec_trajectory_tri_wenfa")


# ------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------
def main():
    print("Loading trajectory data …")
    td = load_trajectories()
    print(f"  {td['num_seqs']} trajectories, lengths={td['lengths']}")

    print("Plotting triangular trajectory comparison (Traj 2–4) …")
    plot_tri_trajectories(td)

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
