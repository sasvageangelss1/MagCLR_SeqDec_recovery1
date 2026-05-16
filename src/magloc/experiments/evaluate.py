from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from magloc.data.datasets import RegressionWindowDataset
from magloc.eval.metrics import localization_metrics, save_metrics
from magloc.eval.retrieval import NumpyRetriever, softmax_weighted_position
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    ckpt = torch.load(regression_ckpt, map_location=device, weights_only=False)
    model = make_model(cfg).to(device)
    model.load_state_dict(ckpt.get("model") or ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    head = RegressionHead(int(cfg["model"].get("embed_dim", 256)), dropout=float(cfg["finetune"].get("dropout", 0.1))).to(device)
    head.load_state_dict(ckpt.get("head") or ckpt.get("reg_head_state_dict")); head.eval()
    pos_mean = ckpt.get("pos_mean"); pos_std = ckpt.get("pos_std")
    preds, gts = [], []
    for x, y in loader:
        pred = head(model(x.to(device), return_proj=False)).cpu().numpy()
        if pos_mean is not None:
            pred = pred * np.asarray(pos_std) + np.asarray(pos_mean)
        preds.append(pred); gts.append(y.numpy())
    pred = np.concatenate(preds); gt = np.concatenate(gts)
    metrics = localization_metrics(pred, gt, jump_threshold_m=float(cfg["evaluation"].get("jump_threshold_m", 2.5)))
    save_metrics(metrics, out / f"{split_name}_regression_metrics.json")
    np.savez_compressed(out / f"{split_name}_regression_preds.npz", pred=pred, gt=gt)
    print(metrics)
    return metrics


def evaluate_retrieval(config_path: str, encoder_ckpt: str, split_name: str = "test", output_dir: str | None = None):
    cfg = load_yaml(config_path)
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "eval_retrieval")
    db_emb, db_pos, _, _ = extract_embeddings(cfg, encoder_ckpt, "train")
    q_emb, gt, lengths, files = extract_embeddings(cfg, encoder_ckpt, split_name)
    ret = NumpyRetriever(metric=cfg["retrieval"].get("metric", "cosine")).fit(db_emb, db_pos)
    res = ret.query(q_emb, k=int(cfg["retrieval"].get("k", 3)))
    pred = softmax_weighted_position(res.scores, res.positions, tau=float(cfg["retrieval"].get("tau", 0.30)))
    metrics = localization_metrics(pred, gt, jump_threshold_m=float(cfg["evaluation"].get("jump_threshold_m", 2.5)))
    save_metrics(metrics, out / f"{split_name}_retrieval_metrics.json")
    np.savez_compressed(out / f"{split_name}_retrieval_candidates.npz", pred=pred, gt=gt, scores=res.scores, positions=res.positions, lengths=np.asarray(lengths))
    lengths_arr = np.asarray(lengths)
    plot_trajectory_comparison(pred, gt, lengths_arr, "Retrieval: Trajectory Comparison", out)
    plot_error_over_time(pred, gt, lengths_arr, "Retrieval: Error Over Time", out)
    plot_cumulative_error(pred, gt, "Retrieval: Cumulative Error Distribution", out)
    print(metrics)
    return metrics


def evaluate_seqdec(config_path: str, encoder_ckpt: str, split_name: str = "test", output_dir: str | None = None):
    cfg = load_yaml(config_path)
    out = ensure_dir(output_dir or Path(cfg["paths"]["output_root"]) / cfg["scene"]["name"] / "eval_seqdec")
    db_emb, db_pos, _, _ = extract_embeddings(cfg, encoder_ckpt, "train")
    q_emb, gt, lengths, files = extract_embeddings(cfg, encoder_ckpt, split_name)
    ret = NumpyRetriever(metric=cfg["retrieval"].get("metric", "cosine")).fit(db_emb, db_pos)
    res = ret.query(q_emb, k=int(cfg["seqdec"].get("k", 3)))
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
    start = 0
    paths, confs = [], []
    for n in lengths:
        if n <= 0:
            continue
        end = start + n
        decoded = viterbi_decode(res.scores[start:end], res.positions[start:end], sd_cfg)
        preds.append(decoded["pred"])
        paths.append(decoded["path"])
        confs.append(decoded["confidence"])
        start = end
    pred = np.concatenate(preds).astype(np.float32)
    metrics = localization_metrics(pred, gt[: len(pred)], jump_threshold_m=float(cfg["evaluation"].get("jump_threshold_m", 2.5)))
    save_metrics(metrics, out / f"{split_name}_seqdec_metrics.json")
    lengths_arr = np.asarray(lengths)
    np.savez_compressed(
        out / f"{split_name}_seqdec_preds.npz",
        pred=pred, gt=gt[: len(pred)], path=np.concatenate(paths),
        confidence=np.concatenate(confs), lengths=lengths_arr,
    )
    plot_trajectory_comparison(pred, gt[: len(pred)], lengths_arr, "SeqDec: Trajectory Comparison", out)
    plot_error_over_time(pred, gt[: len(pred)], lengths_arr, "SeqDec: Error Over Time", out)
    plot_cumulative_error(pred, gt[: len(pred)], "SeqDec: Cumulative Error Distribution", out)
    plot_confidence(np.concatenate(confs), pred, gt[: len(pred)], lengths_arr, "SeqDec: Confidence", out)
    print(metrics)
    return metrics
