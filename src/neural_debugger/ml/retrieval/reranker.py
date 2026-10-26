"""Optional cross-encoder reranking of the top candidates of any retriever."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from neural_debugger.data.schema import Chunk

from .encoders import resolve_device
from .fusion import rrf_scores
from .types import Retriever, SearchHit

logger = logging.getLogger(__name__)


class PairScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 16) -> list[float]: ...

    def fingerprint(self) -> dict[str, Any]: ...


@dataclass
class CrossEncoderScorer:
    hf_id: str
    revision: str | None = None
    max_length: int = 512
    device: str = "auto"
    name: str = ""

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self.name = self.name or self.hf_id
        self.resolved_device = resolve_device(self.device)
        try:
            self._model = CrossEncoder(
                self.hf_id,
                revision=self.revision,
                max_length=self.max_length,
                device=self.resolved_device,
            )
        except (RuntimeError, OSError) as exc:
            if self.resolved_device == "cpu":
                raise
            logger.warning(
                "cross-encoder load on %s failed (%s); using cpu", self.resolved_device, exc
            )
            self.resolved_device = "cpu"
            self._model = CrossEncoder(
                self.hf_id, revision=self.revision, max_length=self.max_length, device="cpu"
            )

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 16) -> list[float]:
        if not pairs:
            return []
        try:
            out = self._model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False)
        except RuntimeError as exc:
            if self.resolved_device == "cpu":
                raise
            logger.warning(
                "cross-encoder failed on %s (%s); retrying on cpu", self.resolved_device, exc
            )
            self._model.model.to("cpu")
            self._model._target_device = "cpu"  # noqa: SLF001
            self.resolved_device = "cpu"
            out = self._model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False)
        return [float(x) for x in out]

    def fingerprint(self) -> dict[str, Any]:
        return {"hf_id": self.hf_id, "revision": self.revision, "max_length": self.max_length}


def passage_text(chunk: Chunk) -> str:
    return chunk.index_text()


class RerankedRetriever:
    """Rerank the top ``top_n`` candidates of ``base``; the rest keep their base order below.

    ``mode='replace'`` orders candidates by cross-encoder score alone; ``mode='rrf'`` fuses the
    cross-encoder ranking with the base ranking (more robust when the reranker domain-mismatches).
    """

    def __init__(
        self,
        base: Retriever,
        scorer: PairScorer,
        chunk_by_id: dict[str, Chunk],
        top_n: int = 30,
        mode: str = "replace",
        k_rrf: int = 60,
        depth: int = 100,
        name: str = "rerank",
    ) -> None:
        if mode not in {"replace", "rrf"}:
            raise ValueError("mode must be 'replace' or 'rrf'")
        self.name, self.base, self.scorer = name, base, scorer
        self.chunk_by_id, self.top_n, self.mode = chunk_by_id, top_n, mode
        self.k_rrf, self.depth = k_rrf, depth

    def search(self, query: str, k: int = 10, framework: str | None = None) -> list[SearchHit]:
        hits = self.base.search(query, max(self.depth, k, self.top_n), framework)
        head, tail = hits[: self.top_n], hits[self.top_n :]
        scores = self.scorer.score(
            [(query, passage_text(self.chunk_by_id[h.chunk_id])) for h in head]
        )
        by_ce = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        if self.mode == "replace":
            order = by_ce
            new_scores = {i: scores[i] for i in order}
        else:
            ce_rank = [head[i].section_id for i in by_ce]
            fused = rrf_scores([[h.section_id for h in head], ce_rank], None, self.k_rrf)
            order = sorted(range(len(head)), key=lambda i: (-fused[head[i].section_id], i))
            new_scores = {i: fused[head[i].section_id] for i in order}
        reranked = [replace(head[i], score=round(new_scores[i], 6)) for i in order] + tail
        return [replace(h, rank=i + 1) for i, h in enumerate(reranked[:k])]
