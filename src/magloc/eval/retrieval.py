from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore


@dataclass
class RetrievalResult:
    scores: np.ndarray      # (Q,K), cosine similarity or negative distance score; larger is better
    indices: np.ndarray     # (Q,K)
    positions: np.ndarray   # (Q,K,2)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class BaseRetriever(Protocol):
    """Protocol that all retrievers must implement."""

    def fit(self, embeddings: np.ndarray, positions: np.ndarray) -> "BaseRetriever": ...
    def query(self, q: np.ndarray, k: int = 3) -> RetrievalResult: ...


class NumpyRetriever:
    """Brute-force retriever using pure NumPy. Suitable for small/medium databases."""
    def __init__(self, metric: str = "cosine"):
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be cosine or l2")
        self.metric = metric
        self.emb: np.ndarray | None = None
        self.pos: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, positions: np.ndarray) -> "NumpyRetriever":
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
            idx = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
            row = np.arange(len(q))[:, None]
            idx = idx[row, np.argsort(-sim[row, idx], axis=1)]
            scores = sim[row, idx]
        else:
            dist2 = ((q[:, None, :] - self.emb[None, :, :]) ** 2).sum(axis=2)
            idx = np.argpartition(dist2, kth=min(k, dist2.shape[1] - 1), axis=1)[:, :k]
            row = np.arange(len(q))[:, None]
            idx = idx[row, np.argsort(dist2[row, idx], axis=1)]
            scores = -np.sqrt(dist2[row, idx])
        return RetrievalResult(scores=scores.astype(np.float32), indices=idx.astype(np.int64), positions=self.pos[idx].astype(np.float32))


class FaissRetriever:
    """Retriever backed by FAISS (Facebook AI Similarity Search)."""
    def __init__(self, metric: str = "cosine"):
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss is not installed. Install it with: pip install faiss-cpu  (or faiss-gpu)")
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be cosine or l2")
        self.metric = metric
        self.pos: np.ndarray | None = None
        self._index: "faiss.Index" | None = None

    def fit(self, embeddings: np.ndarray, positions: np.ndarray) -> "FaissRetriever":
        embeddings = embeddings.astype(np.float32)
        self.pos = positions.astype(np.float32)
        d = embeddings.shape[1]

        if self.metric == "cosine":
            embeddings = l2_normalize(embeddings)
            self._index = faiss.IndexFlatIP(d)
        else:
            self._index = faiss.IndexFlatL2(d)

        self._index.add(embeddings)
        return self

    def query(self, q: np.ndarray, k: int = 3) -> RetrievalResult:
        if self._index is None or self.pos is None:
            raise RuntimeError("fit must be called before query")
        q = q.astype(np.float32)
        if self.metric == "cosine":
            q = l2_normalize(q)
        distances, indices = self._index.search(q, k)
        if self.metric == "cosine":
            scores = distances.astype(np.float32)
        else:
            scores = (-distances).astype(np.float32)
        return RetrievalResult(
            scores=scores,
            indices=indices.astype(np.int64),
            positions=self.pos[indices].astype(np.float32),
        )


def get_retriever(
    backend: Literal["numpy", "faiss"] = "numpy",
    metric: str = "cosine",
) -> BaseRetriever:
    """Factory: return the requested retriever instance.

    Args:
        backend: "numpy" (default) or "faiss".
        metric:  "cosine" (inner-product on L2-normalised vectors) or "l2".

    Returns:
        A retriever instance implementing the BaseRetriever protocol.
    """
    if backend == "faiss":
        return FaissRetriever(metric=metric)
    return NumpyRetriever(metric=metric)


def check_faiss_available() -> bool:
    """Return True if faiss is installed and importable."""
    return _FAISS_AVAILABLE


def softmax_weighted_position(scores: np.ndarray, positions: np.ndarray, tau: float = 0.30) -> np.ndarray:
    s = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(s / max(tau, 1e-6))
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    return (positions * w[:, :, None]).sum(axis=1).astype(np.float32)
