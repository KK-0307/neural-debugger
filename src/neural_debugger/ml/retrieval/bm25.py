"""BM25 (Okapi, Lucene-style non-negative IDF) over corpus chunks.

Deliberately dependency-light and deterministic: the inverted index is built in memory from the
corpus file in ~1 s, ties are broken by chunk order, and results are collapsed to one hit per
section so one long page cannot fill the result list.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from neural_debugger.data.schema import Chunk

from .tokenizer import TokenizerConfig, tokenize
from .types import SearchHit

SNIPPET_CHARS = 280


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    tokenizer: TokenizerConfig = TokenizerConfig()

    def as_dict(self) -> dict[str, object]:
        return {"k1": self.k1, "b": self.b, "tokenizer": self.tokenizer.as_dict()}


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], config: BM25Config = BM25Config()) -> None:
        self.config = config
        self.name = "bm25"
        self.chunks = chunks
        started = time.perf_counter()
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(chunks), dtype=np.float64)
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk.index_text(), config.tokenizer)
            lengths[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term].append((i, tf))
        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) if len(chunks) else 0.0
        n = len(chunks)
        self.idf: dict[str, float] = {}
        self._post_idx: dict[str, np.ndarray] = {}
        self._post_tf: dict[str, np.ndarray] = {}
        for term, plist in postings.items():
            df = len(plist)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            self._post_idx[term] = np.fromiter((p[0] for p in plist), dtype=np.int64, count=df)
            self._post_tf[term] = np.fromiter((p[1] for p in plist), dtype=np.float64, count=df)
        self._frameworks = np.array([c.framework for c in chunks])
        self.build_seconds = time.perf_counter() - started

    def score_chunks(self, query: str) -> np.ndarray:
        """BM25 score of every chunk for ``query`` (0 where no query term occurs)."""
        k1, b = self.config.k1, self.config.b
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        norm = k1 * (1.0 - b + b * self.doc_len / self.avgdl) if self.avgdl else None
        for term in set(tokenize(query, self.config.tokenizer)):
            idx = self._post_idx.get(term)
            if idx is None or norm is None:
                continue
            tf = self._post_tf[term]
            scores[idx] += self.idf[term] * tf * (k1 + 1.0) / (tf + norm[idx])
        return scores

    def search(self, query: str, k: int = 10, framework: str | None = None) -> list[SearchHit]:
        scores = self.score_chunks(query)
        if framework is not None:
            scores = np.where(self._frameworks == framework, scores, 0.0)
        candidates = np.flatnonzero(scores > 0)
        # Sort by score descending; ties broken by corpus order (deterministic).
        order = candidates[np.lexsort((candidates, -scores[candidates]))]
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for i in order:
            chunk = self.chunks[int(i)]
            if chunk.section_id in seen:
                continue
            seen.add(chunk.section_id)
            hits.append(_to_hit(len(hits) + 1, float(scores[i]), chunk))
            if len(hits) >= k:
                break
        return hits


def _to_hit(rank: int, score: float, chunk: Chunk) -> SearchHit:
    body = " ".join(chunk.content.split())
    return SearchHit(
        rank=rank,
        score=round(score, 6),
        chunk_id=chunk.chunk_id,
        section_id=chunk.section_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        heading_path=chunk.heading_path,
        framework=chunk.framework,
        version=chunk.version,
        url=chunk.url,
        source_url=chunk.source_url,
        snippet=body[:SNIPPET_CHARS] + ("…" if len(body) > SNIPPET_CHARS else ""),
    )
