"""Data model shared by the ingestion pipeline, retrieval and evaluation.

Identifier conventions (stable across re-runs, used by evaluation labels):

* ``doc_id``     ``{framework}/{version}/{page path without extension}``,
                 e.g. ``python/3.12/tutorial/errors``
* ``section_id`` ``{doc_id}#{anchor}`` where ``anchor`` is the heading/definition id
                 in the published page (``{doc_id}`` alone for text before the first heading)
* ``chunk_id``   ``{section_id}:{ordinal}``

Relevance labels reference *sections*, not chunks, so they stay valid when the
chunking parameters change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BlockKind = Literal["paragraph", "code", "list", "table"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Block:
    """A unit of page content that chunking never splits unless it is oversized."""

    kind: BlockKind
    text: str
    lang: str | None = None  # code blocks only


@dataclass
class Section:
    anchor: str  # "" for text before the first heading
    heading: str
    heading_path: list[str]
    blocks: list[Block] = field(default_factory=list)


@dataclass
class ParsedPage:
    title: str
    sections: list[Section]


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    title: str
    source_url: str
    framework: str
    version: str
    license: str
    raw_sha256: str  # hash of the exact bytes the parse was made from
    retrieved_at: str  # ISO-8601 UTC; recorded at fetch time, never regenerated
    n_sections: int
    n_chunks: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section_id: str
    ordinal: int
    title: str
    framework: str
    version: str
    source_url: str  # page URL
    url: str  # page URL + "#anchor": the citation link
    section_heading: str
    heading_path: list[str]
    content: str
    content_hash: str
    n_words: int
    ingested_at: str

    def index_text(self) -> str:
        """The text BM25 (and later the dense encoder) sees: page title, headings, body."""
        return f"{self.title}\n{' > '.join(self.heading_path)}\n{self.content}"


def to_json_line(obj: Any) -> str:
    return json.dumps(asdict(obj), ensure_ascii=False, sort_keys=True)


def chunk_from_dict(d: dict[str, Any]) -> Chunk:
    return Chunk(**d)
