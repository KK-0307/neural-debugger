"""Reading the processed corpus."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .schema import Chunk, DocumentRecord, chunk_from_dict, sha256_bytes

DEFAULT_CORPUS_DIR = Path("data/corpus")


@dataclass
class Corpus:
    chunks: list[Chunk]
    documents: list[DocumentRecord]
    sha256: str

    def __post_init__(self) -> None:
        self._by_section: dict[str, list[Chunk]] = defaultdict(list)
        for c in self.chunks:
            self._by_section[c.section_id].append(c)

    @property
    def section_ids(self) -> set[str]:
        return set(self._by_section)

    def section_chunks(self, section_id: str) -> list[Chunk]:
        return self._by_section[section_id]

    def section_text(self, section_id: str) -> str:
        return "\n\n".join(
            c.content for c in sorted(self._by_section[section_id], key=lambda c: c.ordinal)
        )

    def section_url(self, section_id: str) -> str:
        return self._by_section[section_id][0].url

    def section_heading_path(self, section_id: str) -> list[str]:
        return self._by_section[section_id][0].heading_path


def load_corpus(corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> Corpus:
    corpus_dir = Path(corpus_dir)
    raw = (corpus_dir / "chunks.jsonl").read_bytes()
    chunks = [
        chunk_from_dict(json.loads(line)) for line in raw.decode("utf-8").splitlines() if line
    ]
    docs = [
        DocumentRecord(**json.loads(line))
        for line in (corpus_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return Corpus(chunks=chunks, documents=docs, sha256=sha256_bytes(raw))
