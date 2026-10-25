"""Seeded random ranking: the chance-level floor every real retriever must clear."""

from __future__ import annotations

import random

from neural_debugger.data.schema import Chunk

from .bm25 import _to_hit
from .types import SearchHit


class RandomRetriever:
    def __init__(self, chunks: list[Chunk], seed: int = 0) -> None:
        self.name = "random"
        self.chunks = chunks
        self.seed = seed
        self._sections: dict[str, Chunk] = {}
        for c in chunks:
            self._sections.setdefault(c.section_id, c)

    def search(self, query: str, k: int = 10, framework: str | None = None) -> list[SearchHit]:
        # Deterministic per (seed, query) so repeated evaluation gives identical numbers.
        rng = random.Random(f"{self.seed}:{query}")
        pool = [s for s in self._sections.values() if framework is None or s.framework == framework]
        rng.shuffle(pool)
        return [_to_hit(i + 1, 0.0, c) for i, c in enumerate(pool[:k])]
