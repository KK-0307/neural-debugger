"""Shared builders for tests."""

from __future__ import annotations

from neural_debugger.data.schema import Chunk, sha256_text


def make_chunk(
    section: str,
    content: str,
    *,
    framework: str = "python",
    doc: str | None = None,
    ordinal: int = 0,
    title: str = "Page",
    heading: str | None = None,
) -> Chunk:
    doc_id = doc or f"{framework}/1.0/page"
    anchor = section
    section_id = f"{doc_id}#{anchor}"
    return Chunk(
        chunk_id=f"{section_id}:{ordinal}",
        doc_id=doc_id,
        section_id=section_id,
        ordinal=ordinal,
        title=title,
        framework=framework,
        version="1.0",
        source_url=f"https://example.test/{doc_id}",
        url=f"https://example.test/{doc_id}#{anchor}",
        section_heading=heading or anchor,
        heading_path=[heading or anchor],
        content=content,
        content_hash=sha256_text(content),
        n_words=len(content.split()),
        ingested_at="2026-01-01T00:00:00+00:00",
    )
