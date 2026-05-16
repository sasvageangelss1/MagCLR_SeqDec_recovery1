from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SeqDecConfig:
    tau: float = 0.30
    spatial_sigma_m: float = 1.20
    confidence_alpha: float = 0.75
    expected_step_m: float = 1.0
    displacement_sigma_m: float = 0.80
    max_jump_m: float = 2.50
    beta: float = 0.45
    use_confidence: bool = True
    use_displacement: bool = True
    use_jump_suppression: bool = True


def _softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    s = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(s / max(tau, 1e-6))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def observation_confidence(scores: np.ndarray, positions: np.ndarray, cfg: SeqDecConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute confidence from score sharpness and spatial concentration.

    scores: (T,K), positions: (T,K,2)
    """
    w = _softmax(scores, cfg.tau)
    k = scores.shape[1]
    entropy = -(w * np.log(w + 1e-12)).sum(axis=1) / np.log(max(k, 2))
    sharpness = 1.0 - entropy
    center = (positions * w[:, :, None]).sum(axis=1)
    disp = ((positions - center[:, None, :]) ** 2).sum(axis=2)
    spatial_var = (w * disp).sum(axis=1)
    concentration = np.exp(-spatial_var / (2.0 * cfg.spatial_sigma_m ** 2 + 1e-12))
    conf = cfg.confidence_alpha * sharpness + (1.0 - cfg.confidence_alpha) * concentration
    return conf.astype(np.float32), {"weights": w.astype(np.float32), "sharpness": sharpness.astype(np.float32), "concentration": concentration.astype(np.float32)}


def emission_prob(scores: np.ndarray, positions: np.ndarray, cfg: SeqDecConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    w = _softmax(scores, cfg.tau)
    conf, detail = observation_confidence(scores, positions, cfg)
    k = scores.shape[1]
    if not cfg.use_confidence:
        emit = w
        conf = np.ones(scores.shape[0], dtype=np.float32)
    else:
        emit = conf[:, None] * w + (1.0 - conf[:, None]) * (1.0 / k)
    emit = emit / (emit.sum(axis=1, keepdims=True) + 1e-12)
    return emit.astype(np.float32), conf.astype(np.float32), detail


def transition_matrix(prev_pos: np.ndarray, cur_pos: np.ndarray, cfg: SeqDecConfig) -> np.ndarray:
    d = np.linalg.norm(prev_pos[:, None, :] - cur_pos[None, :, :], axis=2)  # (K,K)
    if cfg.use_jump_suppression:
        feasible = d <= cfg.max_jump_m
    else:
        feasible = np.ones_like(d, dtype=bool)
    if cfg.use_displacement:
        cost = ((d - cfg.expected_step_m) ** 2) / (2.0 * cfg.displacement_sigma_m ** 2 + 1e-12)
        trans = np.exp(-cfg.beta * cost) * feasible.astype(np.float32)
    else:
        trans = feasible.astype(np.float32)
    row_sum = trans.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1e-12
    if np.any(bad):
        # fallback: if all candidates are filtered, keep the least-displacement candidate instead of crashing
        trans[bad] = 0.0
        best = np.argmin(d[bad], axis=1)
        trans[np.where(bad)[0], best] = 1.0
        row_sum = trans.sum(axis=1, keepdims=True)
    return (trans / (row_sum + 1e-12)).astype(np.float32)


def viterbi_decode(scores: np.ndarray, positions: np.ndarray, cfg: SeqDecConfig = SeqDecConfig()) -> Dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    if scores.ndim != 2 or positions.ndim != 3:
        raise ValueError("scores must be (T,K), positions must be (T,K,2)")
    t_len, k = scores.shape
    emit, conf, detail = emission_prob(scores, positions, cfg)
    log_emit = np.log(emit + 1e-12)
    dp = np.full((t_len, k), -np.inf, dtype=np.float64)
    back = np.zeros((t_len, k), dtype=np.int64)
    dp[0] = -np.log(k) + log_emit[0]
    for t in range(1, t_len):
        trans = transition_matrix(positions[t - 1], positions[t], cfg)
        log_trans = np.log(trans + 1e-12)
        scores_t = dp[t - 1][:, None] + log_trans
        back[t] = scores_t.argmax(axis=0)
        dp[t] = scores_t.max(axis=0) + log_emit[t]
    path = np.zeros(t_len, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for t in range(t_len - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    pred = positions[np.arange(t_len), path]
    return {
        "path": path,
        "pred": pred.astype(np.float32),
        "confidence": conf,
        "emission": emit,
        "sharpness": detail["sharpness"],
        "concentration": detail["concentration"],
    }
