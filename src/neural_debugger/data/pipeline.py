"""Corpus build: fetch -> parse -> chunk -> de-duplicate -> write (deterministic).

Given the same raw cache and config, ``build_corpus`` produces byte-identical output; nothing
in the output depends on wall-clock time (``retrieved_at`` comes from fetch-time metadata).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunking import ChunkingConfig, chunk_section, n_words
from .config import CorpusConfig, SourceSpec, load_config
from .dedupe import dedupe_chunks
from .fetch import FetchError, PoliteFetcher
from .git_source import ensure_checkout
from .html_parser import parse_sphinx_html
from .md_parser import parse_fastapi_markdown
from .schema import (
    Chunk,
    DocumentRecord,
    ParsedPage,
    sha256_bytes,
    sha256_text,
    to_json_line,
)

logger = logging.getLogger(__name__)
PIPELINE_VERSION = "1"


@dataclass
class RawDocument:
    spec: SourceSpec
    doc_id: str
    source_url: str
    raw_sha256: str
    retrieved_at: str
    raw_bytes: int
    parsed: ParsedPage


def _doc_id(spec: SourceSpec, rel_path: str) -> str:
    stem = rel_path.removesuffix(".html").removesuffix(".md")
    return f"{spec.framework}/{spec.version}/{stem}"


def _fastapi_url(spec: SourceSpec, rel_md: str) -> str:
    assert spec.site_url
    stem = rel_md.removesuffix(".md")
    if stem.endswith("/index"):
        stem = stem[: -len("index")].rstrip("/")
    return spec.site_url + (stem + "/" if stem and stem != "index" else "")


def fetch_and_parse_sphinx(
    spec: SourceSpec, fetcher: PoliteFetcher
) -> tuple[list[RawDocument], list[str]]:
    assert spec.base_url
    docs: list[RawDocument] = []
    failures: list[str] = []
    for rel in spec.pages:
        url = spec.base_url + rel
        try:
            raw = fetcher.fetch(url)
        except FetchError as exc:
            failures.append(str(exc))
            continue
        parsed = parse_sphinx_html(raw.content.decode("utf-8", "replace"))
        docs.append(
            RawDocument(
                spec,
                _doc_id(spec, rel),
                url,
                raw.sha256,
                raw.retrieved_at,
                len(raw.content),
                parsed,
            )
        )
    return docs, failures


def parse_git_markdown(spec: SourceSpec, raw_dir: Path) -> list[RawDocument]:
    checkout, retrieved_at = ensure_checkout(spec, raw_dir / "git" / spec.name)
    assert spec.docs_dir
    docs_root = checkout / spec.docs_dir

    def resolve_include(path: str) -> str | None:
        # Includes are written relative to the docs tree ("../../docs_src/x.py"); the
        # target lives at the repository root.
        target = checkout / path.lstrip("./")
        return target.read_text(encoding="utf-8") if target.is_file() else None

    files: set[Path] = set()
    for pattern in spec.include:
        files.update(p for p in docs_root.glob(pattern) if p.is_file())
    excluded: set[Path] = set()
    for pattern in spec.exclude:
        excluded.update(docs_root.glob(pattern))
    docs: list[RawDocument] = []
    for path in sorted(files - excluded):
        rel = path.relative_to(docs_root).as_posix()
        data = path.read_bytes()
        parsed = parse_fastapi_markdown(data.decode("utf-8"), resolve_include)
        docs.append(
            RawDocument(
                spec,
                _doc_id(spec, rel),
                _fastapi_url(spec, rel),
                sha256_bytes(data),
                retrieved_at,
                len(data),
                parsed,
            )
        )
    return docs


def _url_with_anchor(page_url: str, anchor: str) -> str:
    return f"{page_url}#{anchor}" if anchor else page_url


def make_chunks(doc: RawDocument, config: ChunkingConfig) -> list[Chunk]:
    out: list[Chunk] = []
    seen_sections: set[str] = set()
    for section in doc.parsed.sections:
        section_id = f"{doc.doc_id}#{section.anchor}" if section.anchor else doc.doc_id
        if section_id in seen_sections:  # would make chunk ids collide and labels ambiguous
            raise ValueError(
                f"duplicate section id {section_id!r}: parser emitted two anchorless sections"
            )
        seen_sections.add(section_id)
        for ordinal, body in enumerate(chunk_section(section, config)):
            out.append(
                Chunk(
                    chunk_id=f"{section_id}:{ordinal}",
                    doc_id=doc.doc_id,
                    section_id=section_id,
                    ordinal=ordinal,
                    title=doc.parsed.title,
                    framework=doc.spec.framework,
                    version=doc.spec.version,
                    source_url=doc.source_url,
                    url=_url_with_anchor(doc.source_url, section.anchor),
                    section_heading=section.heading,
                    heading_path=section.heading_path,
                    content=body,
                    content_hash=sha256_text(body),
                    n_words=n_words(body),
                    ingested_at=doc.retrieved_at,
                )
            )
    return out


def build_corpus(
    config_path: Path,
    raw_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    chunking: ChunkingConfig = ChunkingConfig(),
    fetcher: PoliteFetcher | None = None,
) -> dict[str, Any]:
    cfg: CorpusConfig = load_config(config_path)
    own_fetcher = fetcher is None
    fetcher = fetcher or PoliteFetcher(raw_dir, cfg.user_agent, cfg.request_delay_seconds)
    all_docs: list[RawDocument] = []
    failures: list[str] = []
    try:
        for spec in cfg.sources:
            if spec.kind == "sphinx_html":
                docs, failed = fetch_and_parse_sphinx(spec, fetcher)
                failures.extend(failed)
            else:
                docs = parse_git_markdown(spec, raw_dir)
            all_docs.extend(docs)
    finally:
        if own_fetcher:
            fetcher.close()
    if failures:
        raise FetchError(
            "pages failed to fetch (fix configs/sources.yaml):\n" + "\n".join(failures)
        )

    chunks = [c for d in all_docs for c in make_chunks(d, chunking)]
    kept, report = dedupe_chunks(chunks)

    per_doc_chunks: dict[str, int] = {}
    per_doc_sections: dict[str, set[str]] = {}
    for c in kept:
        per_doc_chunks[c.doc_id] = per_doc_chunks.get(c.doc_id, 0) + 1
        per_doc_sections.setdefault(c.doc_id, set()).add(c.section_id)
    records = [
        DocumentRecord(
            doc_id=d.doc_id,
            title=d.parsed.title,
            source_url=d.source_url,
            framework=d.spec.framework,
            version=d.spec.version,
            license=d.spec.license,
            raw_sha256=d.raw_sha256,
            retrieved_at=d.retrieved_at,
            n_sections=len(per_doc_sections.get(d.doc_id, ())),
            n_chunks=per_doc_chunks.get(d.doc_id, 0),
        )
        for d in all_docs
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_bytes = ("\n".join(to_json_line(c) for c in kept) + "\n").encode("utf-8")
    (out_dir / "chunks.jsonl").write_bytes(chunks_bytes)
    (out_dir / "documents.jsonl").write_text(
        "\n".join(to_json_line(r) for r in records) + "\n", encoding="utf-8"
    )

    by_framework: dict[str, dict[str, int]] = {}
    for r in records:
        agg = by_framework.setdefault(r.framework, {"documents": 0, "sections": 0, "chunks": 0})
        agg["documents"] += 1
        agg["sections"] += r.n_sections
        agg["chunks"] += r.n_chunks
    manifest: dict[str, Any] = {
        "corpus_id": cfg.corpus_id,
        "pipeline_version": PIPELINE_VERSION,
        "sources_config_sha256": sha256_bytes(Path(config_path).read_bytes()),
        "chunking": chunking.as_dict(),
        "corpus_chunks_sha256": sha256_bytes(chunks_bytes),
        "totals": {
            "documents": len(records),
            "sections": sum(r.n_sections for r in records),
            "chunks": len(kept),
            "words": sum(c.n_words for c in kept),
        },
        "by_framework": by_framework,
        "sources": [
            {
                "name": s.name,
                "framework": s.framework,
                "version": s.version,
                "kind": s.kind,
                "license": s.license,
                "base_url": s.base_url or s.site_url,
                **(
                    {"repo": s.repo, "ref": s.ref, "commit": s.commit}
                    if s.kind == "git_markdown"
                    else {}
                ),
            }
            for s in cfg.sources
        ],
        "dedup": report.summary(),
        "dedup_dropped": report.dropped,
        "documents": [
            {
                "doc_id": d.doc_id,
                "source_url": d.source_url,
                "raw_sha256": d.raw_sha256,
                "raw_bytes": d.raw_bytes,
                "retrieved_at": d.retrieved_at,
            }
            for d in all_docs
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("corpus built: %s", manifest["totals"])
    return manifest


__all__ = ["build_corpus", "make_chunks", "asdict"]
