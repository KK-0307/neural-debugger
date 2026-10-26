"""Dense (embedding) retriever over corpus chunks, returning section-collapsed hits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neural_debugger.data.schema import Chunk

from .bm25 import _to_hit
from .encoders import Array, Encoder
from .types import SearchHit
from .vector_index import VectorIndex

CANDIDATE_CHUNKS = 200  # chunks fetched before collapsing to sections


class DenseRetriever:
    def __init__(
        self, chunks: list[Chunk], encoder: Encoder, vectors: Array, name: str = "dense"
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one embedding per chunk is required")
        self.name = name
        self.chunks = chunks
        self.encoder = encoder
        self.index = VectorIndex(vectors, [c.framework for c in chunks])

    def encode_query(self, query: str) -> Array:
        return np.asarray(self.encoder.encode_queries([query])[0], dtype=np.float32)

    def search_vector(self, qvec: Array, k: int, framework: str | None) -> list[SearchHit]:
        scores, ids = self.index.search(qvec, max(CANDIDATE_CHUNKS, 4 * k), framework)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for score, i in zip(scores, ids, strict=True):
            chunk = self.chunks[int(i)]
            if chunk.section_id in seen:
                continue
            seen.add(chunk.section_id)
            hits.append(_to_hit(len(hits) + 1, float(score), chunk))
            if len(hits) >= k:
                break
        return hits

    def search(self, query: str, k: int = 10, framework: str | None = None) -> list[SearchHit]:
        return self.search_vector(self.encode_query(query), k, framework)

    def search_many(
        self, queries: Sequence[str], k: int, framework: str | None = None
    ) -> list[list[SearchHit]]:
        vecs = self.encoder.encode_queries(list(queries))
        return [self.search_vector(v, k, framework) for v in vecs]


def cosine(a: Array, b: Array) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
