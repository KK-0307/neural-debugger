from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchHit:
    """One retrieved section (represented by its best-scoring chunk)."""

    rank: int
    score: float
    chunk_id: str
    section_id: str
    doc_id: str
    title: str
    heading_path: list[str]
    framework: str
    version: str
    url: str  # deep link to the section: the citation target
    source_url: str  # page URL
    snippet: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10, framework: str | None = None) -> list[SearchHit]:
        """Top-k *sections* (document-level de-duplication), best first."""
        ...
