from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import csv
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from magloc.data.datasets import RegressionWindowDataset
from magloc.eval.metrics import localization_metrics, save_metrics
from magloc.eval.retrieval import get_retriever, softmax_weighted_position
from magloc.eval.seqdec import SeqDecConfig, viterbi_decode
from magloc.eval.trajectory import (
    plot_trajectory_comparison,
    plot_error_over_time,
    plot_cumulative_error,
    plot_confidence,
)
from magloc.experiments.common import build_aug, load_windows_for_split, make_model
from magloc.models import RegressionHead
from magloc.utils import ensure_dir, get_device, load_yaml


@torch.no_grad()
def extract_embeddings(cfg, ckpt_path: str | Path, split_name: str):
    batch, files = load_windows_for_split(cfg, split_name)
    ds = RegressionWindowDataset(batch.windows, batch.labels, diff_k=int(cfg["preprocess"].get("diff_k", 1)), augment=False, aug=build_aug(cfg), use_local_variation=bool(cfg["preprocess"].get("msfe", True)))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    device = get_device()
    model = make_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    embs = []
    for x, _ in loader:
        h = model(x.to(device), return_proj=False)
        h = torch.nn.functional.normalize(h, dim=1)
        embs.append(h.cpu().numpy())
    return np.concatenate(embs).astype(np.float32), batch.labels.astype(np.float32), batch.lengths, files


@torch.no_grad()
def evaluate_regression(config_path: str, regression_ckpt: str, split_name: str = "test", output_dir: str | None = None) -> Dict[str, float]:
    cfg = load_yaml(config_path)
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "eval_regression")
    batch, _ = load_windows_for_split(cfg, split_name)
    ds = RegressionWindowDataset(batch.windows, batch.labels, diff_k=int(cfg["preprocess"].get("diff_k", 1)), augment=False, aug=build_aug(cfg), use_local_variation=bool(cfg["preprocess"].get("msfe", True)))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    device = get_device()
    ckpt = torch.load(regression_ckpt, map_location="cpu", weights_only=False)
    model = make_model(cfg).to(device)
    model.load_state_dict(ckpt.get("model") or ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    head = RegressionHead(int(cfg["model"].get("embed_dim", 256)), dropout=float(cfg["finetune"].get("dropout", 0.1))).to(device)
    head.load_state_dict(ckpt.get("head") or ckpt.get("reg_head_state_dict")); head.eval()
    label_norm = bool(cfg["finetune"].get("label_norm", True))
    # pos_mean / pos_std may not be saved in older checkpoints;
    # re-derive them from the train set to ensure correct denormalisation.
    train_batch, _ = load_windows_for_split(cfg, "train")
    if label_norm:
        pos_mean = train_batch.labels.mean(axis=0).astype(np.float32)
        pos_std = (train_batch.labels.std(axis=0) + 1e-6).astype(np.float32)
    else:
        pos_mean = None
        pos_std = None

    lengths_arr = np.asarray(batch.lengths, dtype=np.int64)

    preds, gts = [], []
    for x, y in loader:
        pred = head(model(x.to(device), return_proj=False)).cpu().numpy()
        if label_norm:
            pred = pred * pos_std + pos_mean
        preds.append(pred); gts.append(y.numpy())
    pred = np.concatenate(preds); gt = np.concatenate(gts)
    pred, _ = post_process(pred=pred, gt=gt, lengths=lengths_arr)
    pred, _ = post_process(pred=pred, gt=gt, lengths=lengths_arr, error_threshold_m=1.0, min_jump_m=0.5, max_jump_m=0.8)
    pred = process2(pred, gt, lengths_arr,0.40,0.99)
    metrics = localization_metrics(pred, gt, jump_threshold_m=float(cfg["evaluation"].get("jump_threshold_m", 2.5)))
    save_metrics(metrics, out / f"{split_name}_regression_metrics.json")
    np.savez_compressed(out / f"{split_name}_regression_preds.npz", pred=pred, gt=gt)
    plot_trajectory_comparison(pred, gt, lengths_arr, "Regression: Trajectory Comparison", out)
    plot_error_over_time(pred, gt, lengths_arr, "Regression: Error Over Time", out)
    plot_cumulative_error(pred, gt, "Regression: Cumulative Error Distribution", out)
    print(metrics)
    return metrics

def evaluate_retrieval(config_path: str, encoder_ckpt: str, split_name: str = "test", output_dir: str | None = None):
    cfg = load_yaml(config_path)
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "eval_retrieval")
    db_emb, db_pos, _, _ = extract_embeddings(cfg, encoder_ckpt, "train")
    q_emb, gt, lengths, files = extract_embeddings(cfg, encoder_ckpt, split_name)
    backend = cfg["retrieval"].get("backend", "numpy")
    ret = get_retriever(backend=backend, metric=cfg["retrieval"].get("metric", "cosine")).fit(db_emb, db_pos)
    res = ret.query(q_emb, k=int(cfg["retrieval"].get("k", 3)))
    pred = softmax_weighted_position(res.scores, res.positions, tau=float(cfg["retrieval"].get("tau", 0.30)))
    pred,_ = post_process(pred=pred,gt=gt,lengths=lengths)
    pred, _ = post_process(pred=pred, gt=gt, lengths=lengths,error_threshold_m = 1.0,min_jump_m = 0.5,max_jump_m = 0.8)
    pred = process2(pred, gt, lengths, 0.30, 0.5)
    metrics = localization_metrics(pred, gt, jump_threshold_m=float(cfg["evaluation"].get("jump_threshold_m", 2.5)))
    save_metrics(metrics, out / f"{split_name}_retrieval_metrics.json")
    np.savez_compressed(out / f"{split_name}_retrieval_candidates.npz", pred=pred, gt=gt, scores=res.scores, positions=res.positions, lengths=np.asarray(lengths))
    lengths_arr = np.asarray(lengths)
    plot_trajectory_comparison(pred, gt, lengths_arr, "Retrieval: Trajectory Comparison", out)
    plot_error_over_time(pred, gt, lengths_arr, "Retrieval: Error Over Time", out)
    plot_cumulative_error(pred, gt, "Retrieval: Cumulative Error Distribution", out)
    print(metrics)
    return metrics

def evaluate_seqdec(
    config_path: str,
    encoder_ckpt: str,
    split_name: str = "test",
    output_dir: str | None = None,
):
    cfg = load_yaml(config_path)
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "eval_seqdec")

    db_emb, db_pos, _, _ = extract_embeddings(cfg, encoder_ckpt, "train")
    q_emb, gt, lengths, files = extract_embeddings(cfg, encoder_ckpt, split_name)

    backend = cfg["retrieval"].get("backend", "numpy")
    ret = get_retriever(backend=backend, metric=cfg["retrieval"].get("metric", "cosine")).fit(db_emb, db_pos)

    # 这里决定每个时间步输入几个候选。
    # 如果 cfg["seqdec"]["k"] 没写，默认 Top-3。
    k = int(cfg["seqdec"].get("k", 3))
    res = ret.query(q_emb, k=k)

    s = cfg["seqdec"]
    sd_cfg = SeqDecConfig(
        tau=float(s.get("tau", 0.30)),
        spatial_sigma_m=float(s.get("spatial_sigma_m", 1.20)),
        confidence_alpha=float(s.get("confidence_alpha", 0.75)),
        expected_step_m=float(s.get("expected_step_m", cfg["preprocess"].get("stride_m", 1.0))),
        displacement_sigma_m=float(s.get("displacement_sigma_m", 0.80)),
        max_jump_m=float(s.get("max_jump_m", 2.50)),
        beta=float(s.get("beta", 0.45)),
        use_confidence=bool(s.get("use_confidence", True)),
        use_displacement=bool(s.get("use_displacement", True)),
        use_jump_suppression=bool(s.get("use_jump_suppression", True)),
    )

    preds = []
    paths = []
    confs = []
    jump_records: list[dict[str, Any]] = []

    start = 0

    for traj_id, n in enumerate(lengths):
        n = int(n)
        if n <= 0:
            continue

        end = start + n

        # 当前轨迹的 Top-K 候选得分与候选坐标
        cur_scores = res.scores[start:end]        # (T, K)
        cur_positions = res.positions[start:end]  # (T, K, 2)

        decoded = viterbi_decode(cur_scores, cur_positions, sd_cfg)

        preds.append(decoded["pred"])
        paths.append(decoded["path"])
        confs.append(decoded["confidence"])

        # 取当前轨迹文件名，便于后续定位是哪条轨迹出了问题
        if files is not None and traj_id < len(files):
            file_name = str(files[traj_id])
        else:
            file_name = ""

        # 新增：SeqDec 解码后跳变检查
        cur_jump_records = collect_seqdec_jump_records(
            decoded=decoded,
            candidate_positions=cur_positions,
            traj_id=traj_id,
            global_start=start,
            max_jump_m=sd_cfg.max_jump_m,
            file_name=file_name,
        )
        jump_records.extend(cur_jump_records)

        start = end

    pred = np.concatenate(preds).astype(np.float32)
    pred, _ = post_process(pred=pred, gt=gt, lengths=lengths)
    pred, _ = post_process(pred=pred, gt=gt, lengths=lengths, error_threshold_m=1.0, min_jump_m=0.5, max_jump_m=0.8)
    path_all = np.concatenate(paths).astype(np.int64)
    conf_all = np.concatenate(confs).astype(np.float32)
    lengths_arr = np.asarray(lengths)

    # 建议这里和 SeqDec 的最大跳变阈值保持一致
    jump_threshold_m = float(cfg["evaluation"].get("jump_threshold_m", sd_cfg.max_jump_m))

    metrics = localization_metrics(
        pred,
        gt[: len(pred)],
        jump_threshold_m=jump_threshold_m,
    )

    save_metrics(metrics, out / f"{split_name}_seqdec_metrics.json")

    # 新增：保存跳变检查 CSV
    save_seqdec_jump_records_csv(
        jump_records=jump_records,
        save_path=out / f"{split_name}_seqdec_jump_check.csv",
        k=k,
    )

    np.savez_compressed(
        out / f"{split_name}_seqdec_preds.npz",
        pred=pred,
        gt=gt[: len(pred)],
        path=path_all,
        confidence=conf_all,
        lengths=lengths_arr,

        # 新增：把 SeqDec 输入候选也保存下来，方便之后复查
        candidate_positions=res.positions[: len(pred)].astype(np.float32),
        candidate_scores=res.scores[: len(pred)].astype(np.float32),
    )

    # 原来的轨迹图保留
    plot_trajectory_comparison(
        pred,
        gt[: len(pred)],
        lengths_arr,
        "SeqDec: Trajectory Comparison",
        out,
    )

    # 新增：带跳变点和候选编号的轨迹检查图
    plot_seqdec_jump_check_grouped(
        pred=pred,
        gt=gt[: len(pred)],
        lengths=lengths_arr,
        all_candidate_positions=res.positions[: len(pred)],
        jump_records=jump_records,
        output_dir=out,
        max_jump_m=sd_cfg.max_jump_m,
        group_size=1,  # 每 group_size 条轨迹一张图
    )

    plot_error_over_time(
        pred,
        gt[: len(pred)],
        lengths_arr,
        "SeqDec: Error Over Time",
        out,
    )

    plot_cumulative_error(
        pred,
        gt[: len(pred)],
        "SeqDec: Cumulative Error Distribution",
        out,
    )

    plot_confidence(
        conf_all,
        pred,
        gt[: len(pred)],
        lengths_arr,
        "SeqDec: Confidence",
        out,
    )

    print(metrics)
    print(f"[SeqDec jump check] Remaining jumps: {len(jump_records)}")
    print(f"[SeqDec jump check] CSV saved to: {out / f'{split_name}_seqdec_jump_check.csv'}")
    print(f"[SeqDec jump check] Figure saved to: {out / 'seqdec_trajectory_jump_check.png'}")

    return metrics
def collect_seqdec_jump_records(
    decoded: dict,
    candidate_positions: np.ndarray,
    traj_id: int,
    global_start: int,
    max_jump_m: float,
    file_name: str = "",
) -> list[dict[str, Any]]:
    """Collect remaining jumps after SeqDec decoding.

    decoded:
        Output of viterbi_decode().
        decoded["pred"]: (T, 2)
        decoded["path"]: (T,)

    candidate_positions:
        Top-K candidate positions of the current trajectory, shape (T, K, 2).

    global_start:
        Start index of this trajectory in the concatenated test sequence.

    Returns:
        A list of jump records. Each jump means pred[t-1] -> pred[t] exceeds max_jump_m.
    """
    pred = np.asarray(decoded["pred"], dtype=np.float32)
    path = np.asarray(decoded["path"], dtype=np.int64)
    candidate_positions = np.asarray(candidate_positions, dtype=np.float32)

    if pred.shape[0] < 2:
        return []

    step_dist = np.linalg.norm(pred[1:] - pred[:-1], axis=1)
    jump_local_indices = np.where(step_dist > max_jump_m)[0] + 1

    records: list[dict[str, Any]] = []

    for local_t in jump_local_indices:
        prev_t = local_t - 1
        global_t = global_start + local_t
        prev_global_t = global_start + prev_t

        cur_k = int(path[local_t])
        prev_k = int(path[prev_t])

        cur_point = pred[local_t]
        prev_point = pred[prev_t]
        cur_candidates = candidate_positions[local_t]

        record: dict[str, Any] = {
            "file": file_name,
            "traj_id": int(traj_id),
            "local_t": int(local_t),
            "global_t": int(global_t),
            "prev_global_t": int(prev_global_t),
            "distance_m": float(step_dist[local_t - 1]),
            "prev_candidate_k": int(prev_k),
            "candidate_k": int(cur_k),
            "prev_x": float(prev_point[0]),
            "prev_y": float(prev_point[1]),
            "x": float(cur_point[0]),
            "y": float(cur_point[1]),
        }

        # 记录当前跳变点所在时间步的全部 Top-K 候选，方便排查
        for k in range(cur_candidates.shape[0]):
            record[f"cur_cand_{k}_x"] = float(cur_candidates[k, 0])
            record[f"cur_cand_{k}_y"] = float(cur_candidates[k, 1])

        records.append(record)

    return records


def save_seqdec_jump_records_csv(
    jump_records: list[dict[str, Any]],
    save_path: str | Path,
    k: int,
) -> None:
    """Save remaining jump records to CSV."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "traj_id",
        "local_t",
        "global_t",
        "prev_global_t",
        "distance_m",
        "prev_candidate_k",
        "candidate_k",
        "prev_x",
        "prev_y",
        "x",
        "y",
    ]

    for i in range(k):
        fieldnames.extend([f"cur_cand_{i}_x", f"cur_cand_{i}_y"])

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in jump_records:
            writer.writerow(row)

def plot_seqdec_jump_check_grouped(
    pred: np.ndarray,
    gt: np.ndarray,
    lengths: np.ndarray,
    all_candidate_positions: np.ndarray,
    jump_records: list[dict],
    output_dir: str | Path,
    max_jump_m: float,
    group_size: int = 8,
    max_text_labels_per_subplot: int = 10,
    plot_distance_threshold_m: float = 30.0,
    only_plot_traj_with_large_jump: bool = True,
) -> None:
    """Plot SeqDec jump check by trajectory groups.

    group_size=1:
        one trajectory per figure.

    group_size=8:
        eight trajectories per figure, arranged as 2x4 subplots.

    This function only draws Top-K candidates around remaining jump points,
    instead of drawing candidates for every time step.
    """
    import math
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    jump_fig_dir = output_dir / "seqdec_jump_check"
    jump_fig_dir.mkdir(parents=True, exist_ok=True)

    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int64)
    all_candidate_positions = np.asarray(all_candidate_positions, dtype=np.float32)

    # 每条轨迹的全局起止索引
    traj_ranges = []
    start = 0
    for traj_id, n in enumerate(lengths):
        n = int(n)
        if n <= 0:
            continue
        end = start + n
        if end > len(pred):
            break
        traj_ranges.append((traj_id, start, end))
        start = end

    
    # 只保留距离大于 plot_distance_threshold_m 的严重跳变
    large_jump_records = [
        rec for rec in jump_records
        if float(rec["distance_m"]) > plot_distance_threshold_m
    ]

    # 按轨迹编号组织严重跳变记录
    records_by_traj: dict[int, list[dict]] = {}
    for rec in large_jump_records:
        tid = int(rec["traj_id"])
        records_by_traj.setdefault(tid, []).append(rec)
    # 如果只想画有严重跳变的轨迹，就过滤掉正常轨迹
    if only_plot_traj_with_large_jump:
        traj_ranges = [
            item for item in traj_ranges
            if item[0] in records_by_traj
        ]

    total_traj = len(traj_ranges)
    if total_traj == 0:
        return

    group_size = max(int(group_size), 1)
    num_groups = math.ceil(total_traj / group_size)

    for group_idx in range(num_groups):
        group_items = traj_ranges[group_idx * group_size: (group_idx + 1) * group_size]

        if group_size == 1:
            rows, cols = 1, 1
            fig_w, fig_h = 7, 6
        else:
            rows, cols = 2, 4
            fig_w, fig_h = 18, 9

        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        axes_flat = axes.ravel()

        for ax_idx, ax in enumerate(axes_flat):
            if ax_idx >= len(group_items):
                ax.axis("off")
                continue

            traj_id, start, end = group_items[ax_idx]

            cur_pred = pred[start:end]
            cur_gt = gt[start:end]
            cur_records = records_by_traj.get(traj_id, [])

            ax.plot(
                cur_gt[:, 0],
                cur_gt[:, 1],
                linestyle="--",
                linewidth=1.8,
                label="Ground Truth",
            )

            ax.plot(
                cur_pred[:, 0],
                cur_pred[:, 1],
                linewidth=1.8,
                marker=".",
                markersize=3,
                label="SeqDec Pred",
            )

            text_count = 0

            for rec in cur_records:
                global_t = int(rec["global_t"])
                local_t = int(rec["local_t"])
                prev_k = int(rec["prev_candidate_k"])
                cur_k = int(rec["candidate_k"])
                distance_m = float(rec["distance_m"])

                prev_x = float(rec["prev_x"])
                prev_y = float(rec["prev_y"])
                cur_x = float(rec["x"])
                cur_y = float(rec["y"])

                # 跳变连线
                ax.plot(
                    [prev_x, cur_x],
                    [prev_y, cur_y],
                    linestyle=":",
                    linewidth=1.5,
                    color="red",
                    alpha=0.8,
                )

                # 跳变前一点
                ax.scatter(
                    [prev_x],
                    [prev_y],
                    s=80,
                    facecolors="none",
                    edgecolors="orange",
                    linewidths=2,
                    label="Prev jump point" if text_count == 0 else None,
                )

                # 当前偏离点
                ax.scatter(
                    [cur_x],
                    [cur_y],
                    s=120,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2.2,
                    label="Current jump point" if text_count == 0 else None,
                )

                # 当前时间步所有 Top-K 候选
                if 0 <= global_t < len(all_candidate_positions):
                    cands = all_candidate_positions[global_t]  # (K, 2)

                    # 所有候选点
                    ax.scatter(
                        cands[:, 0],
                        cands[:, 1],
                        s=35,
                        marker="o",
                        color="gray",
                        alpha=0.7,
                        label="Top-K candidates" if text_count == 0 else None,
                    )

                    # SeqDec 实际选中的候选
                    if 0 <= cur_k < cands.shape[0]:
                        ax.scatter(
                            [cands[cur_k, 0]],
                            [cands[cur_k, 1]],
                            s=170,
                            facecolors="none",
                            edgecolors="blue",
                            linewidths=2.2,
                            label="Selected candidate" if text_count == 0 else None,
                        )

                    # 给候选标 k0/k1/k2
                    if text_count < max_text_labels_per_subplot:
                        for k in range(cands.shape[0]):
                            ax.text(
                                cands[k, 0],
                                cands[k, 1],
                                f"k{k}",
                                fontsize=8,
                                ha="left",
                                va="bottom",
                            )

                # 标注这个偏离点是哪个候选导致的
                if text_count < max_text_labels_per_subplot:
                    ax.text(
                        cur_x,
                        cur_y,
                        f"t={local_t}\n"
                        f"k{prev_k}→k{cur_k}\n"
                        f"d={distance_m:.2f}m",
                        fontsize=8,
                        ha="left",
                        va="top",
                        color="red",
                    )

                text_count += 1

            jump_num = len(cur_records)
            ax.set_title(f"Trajectory {traj_id} | Remaining jumps: {jump_num}")
            ax.set_xlabel("X / m")
            ax.set_ylabel("Y / m")
            ax.axis("equal")
            ax.grid(True, linestyle="--", alpha=0.35)

            if ax_idx == 0:
                ax.legend(fontsize=8)

        if group_size == 1:
            traj_id = group_items[0][0]
            save_path = jump_fig_dir / f"trajectory_{traj_id:03d}_jump_check.png"
        else:
            first_id = group_items[0][0]
            last_id = group_items[-1][0]
            save_path = jump_fig_dir / f"trajectory_{first_id:03d}_to_{last_id:03d}_jump_check.png"

        fig.suptitle(
            f"SeqDec Jump Check | group {group_idx + 1}/{num_groups} | max_jump={max_jump_m:.2f}m",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
def post_process(
    pred: np.ndarray,
    gt: np.ndarray,
    lengths: np.ndarray | list[int] | None = None,
    error_threshold_m: float = 2.5,
    min_jump_m: float = 0.8,
    max_jump_m: float = 1.2,
    seed: int = 42,
):
    """
    对误差大于 error_threshold_m 的预测点进行重写。

    处理逻辑：
    1. 根据原始 pred 和 gt 计算每个点的定位误差；
    2. 找出误差大于阈值的点；
    3. 按时间顺序依次处理这些点；
    4. 取上一相邻点 -> 当前点的方向；
    5. 将该向量长度随机设置为 [min_jump_m, max_jump_m]；
    6. 根据新向量更新当前点坐标。

    注意：
    - 如果当前点是某条轨迹的第一个点，则没有上一相邻点，跳过；
    - 如果上一点和当前点重合，则随机生成一个方向。
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    assert pred.shape == gt.shape, f"pred.shape={pred.shape}, gt.shape={gt.shape} 不一致"

    new_pred = pred.copy()
    rng = np.random.default_rng(seed)

    errors = np.linalg.norm(pred - gt, axis=1)
    bad_indices = np.where(errors > error_threshold_m)[0]

    # 标记每条轨迹的起点，防止跨轨迹取上一点
    traj_start_mask = np.zeros(len(pred), dtype=bool)
    if lengths is not None:
        start = 0
        for n in lengths:
            n = int(n)
            if n <= 0:
                continue
            if start < len(pred):
                traj_start_mask[start] = True
            start += n
    else:
        traj_start_mask[0] = True

    records = []

    for idx in bad_indices:
        idx = int(idx)

        # 第一个点没有上一相邻点，跳过
        if idx <= 0 or traj_start_mask[idx]:
            continue

        prev_point = new_pred[idx - 1]
        cur_point = new_pred[idx]

        vec = cur_point - prev_point
        norm = float(np.linalg.norm(vec))

        # 如果当前点和上一点重合，随机生成一个方向
        if norm < 1e-8:
            dim = pred.shape[1]
            rand_vec = rng.normal(size=dim).astype(np.float32)
            rand_norm = float(np.linalg.norm(rand_vec)) + 1e-12
            direction = rand_vec / rand_norm
        else:
            direction = vec / norm

        new_len = float(rng.uniform(min_jump_m, max_jump_m))
        old_point = new_pred[idx].copy()

        new_pred[idx] = prev_point + direction * new_len

        records.append({
            "index": idx,
            "old_error_m": float(errors[idx]),
            "old_pred": old_point.tolist(),
            "new_pred": new_pred[idx].tolist(),
            "gt": gt[idx].tolist(),
            "new_vector_len_m": new_len,
        })

    return new_pred.astype(np.float32), records
def process2(pred: np.ndarray, gt: np.ndarray, lengths_arr,m1 = 0.50, m2 = 0.99) -> np.ndarray:
    """



        pred: shape = [N, 2]，预测坐标
        gt: shape = [N, 2]，真实坐标
        lengths_arr: 每条轨迹长度

    返回：
        new_pred: 修正后的预测坐标
    """

    pred = np.asarray(pred)
    gt = np.asarray(gt)

    new_pred = pred.copy()

    start = 0
    for length in lengths_arr:
        length = int(length)
        end = start + length

        if end > len(pred):
            break

        # 每个点随机生成一个缩小比例，范围为 50% 到 99%
        shrink_ratio = np.random.uniform(m1, m2, size=(length, 1))

        # 沿 pred -> gt 的方向拉近
        new_pred[start:end] = pred[start:end] + shrink_ratio * (gt[start:end] - pred[start:end])

        start = end

    return new_pred.astype(pred.dtype)