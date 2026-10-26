"""Persistent, versioned chunk embeddings.

Layout: ``{root}/{slug}/embeddings.npy`` (float32, row i = chunk i of the corpus file) and
``{root}/{slug}/meta.json``. The metadata pins what the vectors mean: corpus hash, encoder
fingerprint (model id, revision, prefixes, max length), text-template version, device and timing.
An index is only reused if all of that matches, so changing the model or the corpus forces a rebuild
instead of silently serving stale vectors.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neural_debugger.data.schema import Chunk

from .encoders import Array, Encoder

logger = logging.getLogger(__name__)
TEXT_TEMPLATE_VERSION = "index_text:v1"  # Chunk.index_text(): title / heading path / content


class StaleEmbeddingsError(RuntimeError):
    pass


@dataclass
class EmbeddingSet:
    vectors: Array
    meta: dict[str, Any]


def embed_chunks(encoder: Encoder, chunks: list[Chunk], batch_size: int = 32) -> Array:
    # Sort by length for efficient batching, then restore corpus order (embeddings are per-text).
    texts = [c.index_text() for c in chunks]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = encoder.encode_documents([texts[i] for i in order], batch_size=batch_size)
    out = np.empty_like(vectors)
    out[np.asarray(order)] = vectors
    return out


def _expected(encoder: Encoder, corpus_sha: str, n_chunks: int) -> dict[str, Any]:
    return {
        "corpus_chunks_sha256": corpus_sha,
        "n_chunks": n_chunks,
        "text_template": TEXT_TEMPLATE_VERSION,
        "encoder": encoder.fingerprint(),
    }


def _mismatches(meta: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    bad = []
    for key in ("corpus_chunks_sha256", "n_chunks", "text_template"):
        if meta.get(key) != expected[key]:
            bad.append(key)
    mine, want = meta.get("encoder", {}), expected["encoder"]
    for key in want:
        if key == "library_versions":
            continue  # informational; vectors are still comparable across patch versions
        if mine.get(key) != want[key]:
            bad.append(f"encoder.{key}")
    return bad


def load_embeddings(
    directory: Path, encoder: Encoder, corpus_sha: str, n_chunks: int
) -> EmbeddingSet:
    meta = json.loads((directory / "meta.json").read_text())
    bad = _mismatches(meta, _expected(encoder, corpus_sha, n_chunks))
    if bad:
        raise StaleEmbeddingsError(
            f"{directory}: stored embeddings do not match ({', '.join(bad)})"
        )
    vectors = np.load(directory / "embeddings.npy")
    if vectors.shape != (n_chunks, encoder.dim) or vectors.dtype != np.float32:
        raise StaleEmbeddingsError(f"{directory}: unexpected array {vectors.shape} {vectors.dtype}")
    return EmbeddingSet(vectors, meta)


def ensure_embeddings(
    directory: Path,
    encoder: Encoder,
    chunks: list[Chunk],
    corpus_sha: str,
    batch_size: int = 32,
    rebuild: bool = False,
) -> EmbeddingSet:
    """Load matching stored embeddings, or (re)generate and persist them."""
    if not rebuild and (directory / "meta.json").exists():
        try:
            return load_embeddings(directory, encoder, corpus_sha, len(chunks))
        except StaleEmbeddingsError as exc:
            logger.warning("rebuilding embeddings: %s", exc)
    started = time.perf_counter()
    vectors = embed_chunks(encoder, chunks, batch_size)
    seconds = time.perf_counter() - started
    meta = {
        **_expected(encoder, corpus_sha, len(chunks)),
        "embedding_seconds": round(seconds, 2),
        "device": getattr(encoder, "resolved_device", "n/a"),
        "batch_size": batch_size,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "bytes": int(vectors.nbytes),
    }
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", vectors)
    (directory / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return EmbeddingSet(vectors, meta)
