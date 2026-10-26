"""Reciprocal rank fusion (Cormack et al., 2009) over section-level rankings.

``score(s) = sum_i  w_i / (k + rank_i(s))`` with ranks starting at 1; a section missing from a
ranking contributes nothing from it. RRF needs no score calibration between BM25 (unbounded) and
cosine similarity (bounded), which is why it is the default fusion here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .types import Retriever, SearchHit


def rrf_scores(
    rankings: Sequence[Sequence[str]], weights: Sequence[float] | None = None, k: int = 60
) -> dict[str, float]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, section in enumerate(ranking, start=1):
            scores[section] = scores.get(section, 0.0) + w / (k + rank)
    return scores


def fuse_hits(
    hit_lists: Sequence[Sequence[SearchHit]],
    weights: Sequence[float] | None = None,
    k_rrf: int = 60,
    top: int | None = None,
) -> list[SearchHit]:
    """Fuse ranked hit lists; the representative chunk of a section comes from the list that ranks
    that section best (ties: earlier list). Output ranks/scores are renumbered."""
    scores = rrf_scores([[h.section_id for h in hl] for hl in hit_lists], weights, k_rrf)
    best: dict[str, tuple[int, int, SearchHit]] = {}
    for li, hl in enumerate(hit_lists):
        for h in hl:
            cur = best.get(h.section_id)
            if cur is None or (h.rank, li) < (cur[0], cur[1]):
                best[h.section_id] = (h.rank, li, h)
    # score descending; ties by best component rank then list order (deterministic)
    order = sorted(scores, key=lambda s: (-scores[s], best[s][0], best[s][1], s))
    fused = [
        replace(best[s][2], rank=i + 1, score=round(scores[s], 8)) for i, s in enumerate(order)
    ]
    return fused[:top] if top is not None else fused


class HybridRetriever:
    def __init__(
        self,
        retrievers: Sequence[Retriever],
        weights: Sequence[float] | None = None,
        k_rrf: int = 60,
        depth: int = 100,
        name: str = "hybrid",
    ) -> None:
        self.name = name
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        self.k_rrf = k_rrf
        self.depth = depth

    def search(self, query: str, k: int = 10, framework: str | None = None) -> list[SearchHit]:
        depth = max(self.depth, k)
        lists = [r.search(query, depth, framework) for r in self.retrievers]
        return fuse_hits(lists, self.weights, self.k_rrf, top=k)
