"""Shared trajectory visualization utilities for retrieval and seqdec evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_comparison(
    pred: np.ndarray,
    gt: np.ndarray,
    lengths: np.ndarray | None = None,
    title: str = "Trajectory Comparison",
    out_path: Path | str | None = None,
    max_seqs: int = 8,
    dpi: int = 150,
) -> list[plt.Figure]:
    """Plot ground-truth vs predicted trajectories, one subplot per sequence.

    Parameters
    ----------
    pred   : (N, 2) predicted positions in metres.
    gt     : (N, 2) ground-truth positions in metres.
    lengths: per-sequence lengths; if None the whole array is treated as one seq.
    title  : figure title.
    out_path: if given, saves ``trajectory_comparison.png`` there.
    max_seqs: max number of sequence subplots to show in one figure.
    dpi    : figure resolution.

    Returns
    -------
    List of ``matplotlib.figure.Figure`` objects created.
    """
    if lengths is None or len(lengths) == 0:
        lengths = np.array([len(pred)])

    seqs = _split_by_lengths(pred, lengths)
    gt_seqs = _split_by_lengths(gt, lengths)

    figs = []
    cols, rows = 4, 2
    capacity = cols * rows
    buf_pred, buf_gt = [], []

    for i, (p, g) in enumerate(zip(seqs, gt_seqs)):
        if len(p) == 0 or len(g) == 0:
            continue
        buf_pred.append(p)
        buf_gt.append(g)

        if len(buf_pred) == capacity or i == len(seqs) - 1:
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows), squeeze=False)
            fig.suptitle(title, fontsize=14, fontweight="bold")
            for ax, pp, gg in zip(axes.flat, buf_pred, buf_gt):
                ax.plot(gg[:, 0], gg[:, 1], "b-", linewidth=1.5, label="Ground Truth", alpha=0.8)
                ax.plot(pp[:, 0], pp[:, 1], "r--", linewidth=1.5, label="Predicted", alpha=0.8)
                ax.plot(gg[0, 0], gg[0, 1], "bo", markersize=5, label="Start", alpha=0.7)
                ax.plot(gg[-1, 0], gg[-1, 1], "bs", markersize=5, label="End", alpha=0.7)
                ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
                ax.legend(fontsize=7, loc="best")
                ax.set_title(f"Seq {i + 1 - len(buf_pred) + 1}  len={len(pp)}")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.25)
            for ax in axes.flat[len(buf_pred):]:
                ax.axis("off")
            fig.tight_layout()
            if out_path:
                fig.savefig(Path(out_path) / "trajectory_comparison.png", dpi=dpi)
            figs.append(fig)
            plt.close(fig)
            buf_pred, buf_gt = [], []

    return figs


def plot_error_over_time(
    pred: np.ndarray,
    gt: np.ndarray,
    lengths: np.ndarray | None = None,
    title: str = "Localization Error Over Time",
    out_path: Path | str | None = None,
    max_seqs: int = 8,
    dpi: int = 150,
) -> list[plt.Figure]:
    """Plot per-step L2 error as a function of time index, one subplot per sequence.

    Parameters
    ----------
    pred, gt, lengths : see ``plot_trajectory_comparison``.
    title  : figure title.
    out_path: if given, saves ``error_over_time.png`` there.
    max_seqs: see ``plot_trajectory_comparison``.
    dpi    : figure resolution.

    Returns
    -------
    List of ``matplotlib.figure.Figure`` objects created.
    """
    if lengths is None or len(lengths) == 0:
        lengths = np.array([len(pred)])

    seqs = _split_by_lengths(pred, lengths)
    gt_seqs = _split_by_lengths(gt, lengths)

    cols, rows = 4, 2
    capacity = cols * rows
    buf_pairs = []
    figs = []

    for i, (p, g) in enumerate(zip(seqs, gt_seqs)):
        if len(p) == 0 or len(g) == 0:
            continue
        buf_pairs.append((i + 1, p, g))

        if len(buf_pairs) == capacity or i == len(seqs) - 1:
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows), squeeze=False)
            fig.suptitle(title, fontsize=14, fontweight="bold")
            for ax, (seq_idx, pp, gg) in zip(axes.flat, buf_pairs):
                err = np.linalg.norm(pp - gg, axis=1)
                ax.plot(err, color="#E53935", linewidth=1.2, label="L2 Error")
                ax.axhline(err.mean(), color="orange", linestyle="--", linewidth=1, label=f"Mean {err.mean():.2f}m")
                ax.fill_between(range(len(err)), err, alpha=0.2, color="#E53935")
                ax.set_xlabel("Step"); ax.set_ylabel("Error (m)")
                ax.set_title(f"Seq {seq_idx}  mean={err.mean():.2f}m  p90={np.percentile(err, 90):.2f}m")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.25)
            for ax in axes.flat[len(buf_pairs):]:
                ax.axis("off")
            fig.tight_layout()
            if out_path:
                fig.savefig(Path(out_path) / "error_over_time.png", dpi=dpi)
            figs.append(fig)
            plt.close(fig)
            buf_pairs = []

    return figs


def plot_cumulative_error(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str = "Cumulative Error Distribution",
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot empirical CDF of per-step L2 errors.

    Parameters
    ----------
    pred, gt : see ``plot_trajectory_comparison``.
    title   : figure title.
    out_path: if given, saves ``cumulative_error.png`` there.
    dpi     : figure resolution.

    Returns
    -------
    ``matplotlib.figure.Figure``.
    """
    err = np.linalg.norm(pred - gt, axis=1)
    sorted_err = np.sort(err)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_err, cdf, linewidth=2, color="#1565C0")
    ax.fill_between(sorted_err, cdf, alpha=0.15, color="#1565C0")

    for pct, label in [(50, "Median"), (75, "P75"), (90, "P90"), (95, "P95")]:
        xv = np.percentile(err, pct)
        ax.axvline(xv, linestyle="--", linewidth=1, alpha=0.7, label=f"{label}={xv:.2f}m")
        ax.axhline(pct / 100, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Error (m)"); ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(Path(out_path) / "cumulative_error.png", dpi=dpi)
    plt.close(fig)
    return fig


def plot_confidence(
    confidence: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    lengths: np.ndarray | None = None,
    title: str = "SeqDec Confidence",
    out_path: Path | str | None = None,
    max_seqs: int = 8,
    dpi: int = 150,
) -> list[plt.Figure]:
    """Plot per-step confidence alongside the L2 error (SeqDec only).

    Parameters
    ----------
    confidence : (N,) float array of confidence values [0, 1].
    pred, gt  : (N, 2) positions.
    lengths    : per-sequence lengths.
    title      : figure title.
    out_path   : if given, saves ``confidence.png`` there.
    max_seqs   : see ``plot_trajectory_comparison``.
    dpi        : figure resolution.

    Returns
    -------
    List of ``matplotlib.figure.Figure`` objects created.
    """
    if lengths is None or len(lengths) == 0:
        lengths = np.array([len(pred)])

    conf_seqs = _split_by_lengths(confidence, lengths)
    seqs = _split_by_lengths(pred, lengths)
    gt_seqs = _split_by_lengths(gt, lengths)

    cols, rows = 4, 2
    capacity = cols * rows
    buf = []
    figs = []

    for i, (c_seq, p_seq, g_seq) in enumerate(zip(conf_seqs, seqs, gt_seqs)):
        if len(c_seq) == 0:
            continue
        buf.append((i + 1, c_seq, p_seq, g_seq))

        if len(buf) == capacity or i == len(conf_seqs) - 1:
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows), squeeze=False)
            fig.suptitle(title, fontsize=14, fontweight="bold")
            for ax, (seq_idx, c_s, p_s, g_s) in zip(axes.flat, buf):
                err = np.linalg.norm(p_s - g_s, axis=1)
                ax2 = ax.twinx()
                ax.bar(range(len(c_s)), c_s, alpha=0.35, color="#1E88E5", label="Confidence")
                ax2.plot(range(len(err)), err, color="#E53935", linewidth=1.2, label="L2 Error")
                ax2.axhline(err.mean(), color="#FF8F00", linestyle="--", linewidth=1)
                ax.set_xlabel("Step"); ax.set_ylabel("Confidence", color="#1E88E5")
                ax2.set_ylabel("Error (m)", color="#E53935")
                ax.set_title(f"Seq {seq_idx}  mean_err={err.mean():.2f}m  mean_conf={c_s.mean():.2f}")
                ax.set_ylim(0, 1.1); ax2.set_ylim(bottom=0)
                ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
                ax.grid(True, alpha=0.25)
            for ax in axes.flat[len(buf):]:
                ax.axis("off")
            fig.tight_layout()
            if out_path:
                fig.savefig(Path(out_path) / "confidence.png", dpi=dpi)
            figs.append(fig)
            plt.close(fig)
            buf = []

    return figs


def _split_by_lengths(arr: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Split a (N, ...) array into a list of sub-arrays according to lengths."""
    result = []
    pos = 0
    for ln in lengths:
        ln = int(ln)
        if ln <= 0:
            result.append(np.array([]))
            continue
        result.append(arr[pos:pos + ln].copy())
        pos += ln
    return result
