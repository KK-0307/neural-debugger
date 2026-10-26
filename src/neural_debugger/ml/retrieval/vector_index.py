"""Exact inner-product vector search with per-framework sub-indexes.

Two interchangeable exact backends:

* ``numpy``: one matrix-vector product + a deterministic sort (ties -> lower row id).
* ``faiss``: ``IndexFlatIP``. Measured on this corpus (4,995 x 384, M4 Max) FAISS flat search takes
  ~0.14 ms/query vs ~0.06 ms for NumPy, so FAISS brings no speed-up at this scale; it is the seam
  for a larger corpus or an approximate index (HNSW/IVF).

``backend="auto"`` (default) uses FAISS only when it is *safe*: on macOS, FAISS and PyTorch each
bundle a ``libomp`` and the process aborts (``OMP: Error #15``) once both are loaded and FAISS
searches, which would break the CPU-encoding fallback. So auto = FAISS when PyTorch is not loaded
(or off macOS), otherwise NumPy. FAISS is imported lazily for that reason.

Vectors are L2-normalised, so inner product == cosine similarity.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float32]
Backend = str  # "auto" | "numpy" | "faiss"


def faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None


def faiss_is_safe() -> bool:
    """FAISS alongside PyTorch aborts on macOS (duplicate libomp); elsewhere it is fine."""
    return faiss_available() and not (sys.platform == "darwin" and "torch" in sys.modules)


class _NumpyFlat:
    def __init__(self, vectors: Array) -> None:
        self.vectors = vectors

    def search(self, query: Array, k: int) -> tuple[Array, NDArray[np.int64]]:
        scores = self.vectors @ query
        k = min(k, len(scores))
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]  # score desc, then row id
        return scores[order], order.astype(np.int64)


class _FaissFlat:
    def __init__(self, vectors: Array) -> None:
        import faiss  # lazy: see module docstring

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: Array, k: int) -> tuple[Array, NDArray[np.int64]]:
        scores, ids = self.index.search(query.reshape(1, -1), min(k, self.index.ntotal))
        valid = ids[0] >= 0
        return scores[0][valid], ids[0][valid].astype(np.int64)


class VectorIndex:
    def __init__(self, vectors: Array, groups: Sequence[str], backend: Backend = "auto") -> None:
        if vectors.ndim != 2 or len(vectors) != len(groups):
            raise ValueError("vectors must be (n, dim) with one group label per row")
        if backend == "auto":
            backend = "faiss" if faiss_is_safe() else "numpy"
        if backend not in {"numpy", "faiss"}:
            raise ValueError("backend must be 'auto', 'numpy' or 'faiss'")
        if backend == "faiss" and not faiss_available():
            raise RuntimeError("faiss is not installed (install the `index` or `neural` extra)")
        self.backend = backend
        self.dim, self.n = vectors.shape[1], len(vectors)
        labels = np.asarray(list(groups))
        self._ids: dict[str | None, NDArray[np.int64]] = {None: np.arange(self.n, dtype=np.int64)}
        for g in sorted(set(groups)):
            self._ids[g] = np.flatnonzero(labels == g).astype(np.int64)
        cls: Any = _FaissFlat if backend == "faiss" else _NumpyFlat
        self._indexes = {g: cls(np.ascontiguousarray(vectors[ids])) for g, ids in self._ids.items()}

    def search(
        self, query: Array, k: int, group: str | None = None
    ) -> tuple[Array, NDArray[np.int64]]:
        """Top-k (scores, row ids) for one query vector, optionally restricted to a group."""
        if group not in self._indexes:
            raise KeyError(f"unknown group {group!r}")
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
        scores, local = self._indexes[group].search(q, k)
        return scores, self._ids[group][local]
