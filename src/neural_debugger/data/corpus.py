"""Reading the processed corpus."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
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


def load_corpus_overlay(corpus_dirs: Sequence[str | Path]) -> Corpus:
    """Merge several corpus directories in memory (a frozen base plus supplements), in the order given.

    Nothing on disk changes: the base corpus, its manifest, labels and embeddings stay as they were, so earlier
    milestones' results remain reproducible. Section ids must be unique across directories (a page lives in exactly
    one corpus); a clash raises instead of silently shadowing a frozen section.
    """
    if len(corpus_dirs) == 1:
        return load_corpus(corpus_dirs[0])
    chunks: list[Chunk] = []
    docs: list[DocumentRecord] = []
    seen: set[str] = set()
    digest = b""
    for d in corpus_dirs:
        part = load_corpus(d)
        clash = seen & part.section_ids
        if clash:
            raise ValueError(
                f"{d} redefines sections of an earlier corpus, e.g. {sorted(clash)[0]}"
            )
        seen |= part.section_ids
        chunks += part.chunks
        docs += part.documents
        digest += part.sha256.encode()
    return Corpus(chunks=chunks, documents=docs, sha256=sha256_bytes(digest))
