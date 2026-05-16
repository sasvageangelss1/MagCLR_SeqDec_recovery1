from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RetrievalResult:
    scores: np.ndarray      # (Q,K), cosine similarity or negative distance score; larger is better
    indices: np.ndarray     # (Q,K)
    positions: np.ndarray   # (Q,K,2)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class NumpyRetriever:
    """Small/medium-data retriever. Use FAISS only as an optional acceleration, not as a thesis dependency."""
    def __init__(self, metric: str = "cosine"):
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be cosine or l2")
        self.metric = metric
        self.emb = None
        self.pos = None

    def fit(self, embeddings: np.ndarray, positions: np.ndarray):
        self.emb = embeddings.astype(np.float32)
        self.pos = positions.astype(np.float32)
        if self.metric == "cosine":
            self.emb = l2_normalize(self.emb)
        return self

    def query(self, q: np.ndarray, k: int = 3) -> RetrievalResult:
        if self.emb is None or self.pos is None:
            raise RuntimeError("fit must be called before query")
        q = q.astype(np.float32)
        if self.metric == "cosine":
            qn = l2_normalize(q)
            sim = qn @ self.emb.T
            idx = np.argpartition(-sim, kth=min(k, sim.shape[1]-1), axis=1)[:, :k]
            row = np.arange(len(q))[:, None]
            idx = idx[row, np.argsort(-sim[row, idx], axis=1)]
            scores = sim[row, idx]
        else:
            dist2 = ((q[:, None, :] - self.emb[None, :, :]) ** 2).sum(axis=2)
            idx = np.argpartition(dist2, kth=min(k, dist2.shape[1]-1), axis=1)[:, :k]
            row = np.arange(len(q))[:, None]
            idx = idx[row, np.argsort(dist2[row, idx], axis=1)]
            scores = -np.sqrt(dist2[row, idx])
        return RetrievalResult(scores=scores.astype(np.float32), indices=idx.astype(np.int64), positions=self.pos[idx].astype(np.float32))


def softmax_weighted_position(scores: np.ndarray, positions: np.ndarray, tau: float = 0.30) -> np.ndarray:
    s = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(s / max(tau, 1e-6))
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    return (positions * w[:, :, None]).sum(axis=1).astype(np.float32)
