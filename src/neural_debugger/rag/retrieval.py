"""Adapter from the project's retrievers (BM25 / hybrid / custom) to the RAG pipeline's section interface."""

from __future__ import annotations

from typing import Any, Protocol

from neural_debugger.data.corpus import Corpus

from .types import RetrievedSection


class SectionRetriever(Protocol):
    def search(self, query: str, k: int, framework: str | None) -> list[RetrievedSection]: ...


class CorpusSectionRetriever:
    """Wrap any ``ml.retrieval`` retriever and attach each hit's full section text from the corpus.

    Hits whose section is missing from the corpus are skipped: a citation can only point at text that
    really is in the index.
    """

    def __init__(self, retriever: Any, corpus: Corpus) -> None:
        self._retriever = retriever
        self._corpus = corpus
        self._known = corpus.section_ids
        self.name = getattr(retriever, "name", "retriever")

    def search(self, query: str, k: int, framework: str | None) -> list[RetrievedSection]:
        out: list[RetrievedSection] = []
        for hit in self._retriever.search(query, k=k, framework=framework):
            if hit.section_id not in self._known:
                continue
            out.append(
                RetrievedSection(
                    section_id=hit.section_id,
                    rank=hit.rank,
                    score=float(hit.score),
                    framework=hit.framework,
                    version=hit.version,
                    title=hit.title,
                    heading=" > ".join(hit.heading_path),
                    url=self._corpus.section_url(hit.section_id),
                    text=self._corpus.section_text(hit.section_id),
                )
            )
        return out
