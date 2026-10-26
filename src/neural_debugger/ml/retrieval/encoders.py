"""Text encoders for dense retrieval.

``SentenceTransformerEncoder`` wraps a pretrained sentence-transformers model with an explicit,
recorded configuration (HF id, pinned revision, prefixes, max length, device). ``HashingEncoder``
is a small *real* encoder (feature-hashed word uni/bi-grams) with no model download; it exists so
the index, fusion and evaluation plumbing can be tested offline and in CI, not as a quality baseline.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
Array = NDArray[np.float32]


class Encoder(Protocol):
    name: str
    dim: int

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> Array: ...

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> Array: ...

    def fingerprint(self) -> dict[str, Any]:
        """Everything that determines the embedding values (stored next to the embeddings)."""
        ...


def resolve_device(preference: str = "auto") -> str:
    """``auto`` -> cuda, then Apple MPS, then cpu. An explicit device is returned unchanged."""
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2(x: Array) -> Array:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


@dataclass
class SentenceTransformerEncoder:
    hf_id: str
    revision: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq_length: int | None = None
    device: str = "auto"
    name: str = ""

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = self.name or self.hf_id
        self.resolved_device = resolve_device(self.device)
        self._model = self._load(SentenceTransformer, self.resolved_device)
        if self.max_seq_length:
            self._model.max_seq_length = self.max_seq_length
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim() or 0)

    def _load(self, cls: Any, device: str) -> Any:
        try:
            return cls(self.hf_id, revision=self.revision, device=device)
        except (RuntimeError, OSError) as exc:
            if device == "cpu":
                raise
            logger.warning(
                "could not load %s on %s (%s); falling back to cpu", self.hf_id, device, exc
            )
            self.resolved_device = "cpu"
            return cls(self.hf_id, revision=self.revision, device="cpu")

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int) -> Array:
        batch = [prefix + t for t in texts]
        try:
            out = self._model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except RuntimeError as exc:  # e.g. MPS out-of-memory / unsupported op: retry on cpu
            if self.resolved_device == "cpu":
                raise
            logger.warning(
                "encode failed on %s (%s); falling back to cpu", self.resolved_device, exc
            )
            self._model = self._model.to("cpu")
            self.resolved_device = "cpu"
            out = self._model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return _l2(np.asarray(out, dtype=np.float32))

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> Array:
        return self._encode(texts, self.doc_prefix, batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> Array:
        return self._encode(texts, self.query_prefix, batch_size)

    def fingerprint(self) -> dict[str, Any]:
        import sentence_transformers
        import torch

        return {
            "kind": "sentence-transformers",
            "hf_id": self.hf_id,
            "revision": self.revision,
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
            "max_seq_length": int(self._model.max_seq_length),
            "dim": self.dim,
            "normalized": True,
            "library_versions": {
                "sentence_transformers": sentence_transformers.__version__,
                "torch": torch.__version__,
            },
        }


_TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass
class HashingEncoder:
    """Feature-hashed bag of word uni/bi-grams, L2-normalised. Deterministic, dependency-free."""

    dim: int = 256
    name: str = "hashing-encoder"

    def _vec(self, text: str) -> Array:
        v = np.zeros(self.dim, dtype=np.float32)
        words = _TOKEN.findall(text.lower())
        for gram in [*words, *(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))]:
            h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def _encode(self, texts: Sequence[str]) -> Array:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._vec(t) for t in texts]))

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> Array:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> Array:
        return self._encode(texts)

    def fingerprint(self) -> dict[str, Any]:
        return {"kind": "hashing", "dim": self.dim, "normalized": True}
