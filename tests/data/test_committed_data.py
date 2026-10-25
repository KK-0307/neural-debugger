"""Integrity tests on the committed corpus, labels and manifests (the real Milestone 2 artifacts)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.dataset import materialise
from neural_debugger.data.leakage import check_dataset
from neural_debugger.data.queries import load_authored, load_materialised
from neural_debugger.data.schema import sha256_bytes, sha256_text
from neural_debugger.ml.retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data/corpus/chunks.jsonl").exists(), reason="corpus not built"
)
URL_PREFIXES = {
    "python": "https://docs.python.org/3.12/",
    "pytorch": "https://docs.pytorch.org/docs/2.14/",
    "fastapi": "https://fastapi.tiangolo.com/",
}


@pytest.fixture(scope="module")
def corpus():  # type: ignore[no-untyped-def]
    return load_corpus(ROOT / "data/corpus")


def test_corpus_hash_matches_manifest(corpus) -> None:  # type: ignore[no-untyped-def]
    manifest = json.loads((ROOT / "data/manifests/corpus_manifest.json").read_text())
    assert (
        manifest["corpus_chunks_sha256"]
        == corpus.sha256
        == sha256_bytes((ROOT / "data/corpus/chunks.jsonl").read_bytes())
    )
    assert manifest["totals"]["chunks"] == len(corpus.chunks)
    assert manifest["totals"]["documents"] == len(corpus.documents)


def test_every_chunk_has_complete_provenance_and_consistent_ids(corpus) -> None:  # type: ignore[no-untyped-def]
    ids = [c.chunk_id for c in corpus.chunks]
    assert len(ids) == len(set(ids))
    doc_ids = {d.doc_id for d in corpus.documents}
    for c in corpus.chunks:
        assert c.doc_id in doc_ids
        assert c.chunk_id == f"{c.section_id}:{c.ordinal}"
        assert c.section_id.startswith(c.doc_id)
        assert c.content_hash == sha256_text(c.content)
        assert c.content.strip() and c.title and c.version and c.ingested_at
        assert c.framework in URL_PREFIXES


def test_every_citation_url_is_well_formed_and_matches_its_section(corpus) -> None:  # type: ignore[no-untyped-def]
    for c in corpus.chunks:
        assert c.url.startswith(URL_PREFIXES[c.framework]), c.url
        assert re.fullmatch(r"https://[\w.\-/%]+(\.html|/)?(#[^\s#]+)?", c.url), c.url
        anchor = c.section_id.partition("#")[2]
        assert c.url == (f"{c.source_url}#{anchor}" if anchor else c.source_url)


def test_real_citation_urls_were_verified_against_the_published_pages() -> None:
    report = json.loads((ROOT / "data/manifests/url_verification.json").read_text())
    assert set(report) == {"python", "pytorch", "fastapi"}
    for framework, r in report.items():
        assert r["pages"] == r["pages_ok"] > 0, framework
        assert r["anchors"] == r["anchors_found"] > 0, (framework, r["missing_anchors"][:3])


def test_code_blocks_survive_chunking(corpus) -> None:  # type: ignore[no-untyped-def]
    fenced = [c for c in corpus.chunks if "```" in c.content]
    assert len(fenced) > 500
    for c in fenced:
        assert c.content.count("```") % 2 == 0, c.chunk_id  # fences always balanced


def test_materialised_queries_are_in_sync_with_authored_sources(corpus) -> None:  # type: ignore[no-untyped-def]
    fresh = materialise(load_authored(ROOT / "data/eval"))
    assert fresh == (ROOT / "data/eval/queries.jsonl").read_bytes(), (
        "run: python -m neural_debugger.data.dataset build"
    )
    manifest = json.loads((ROOT / "data/manifests/eval_dataset_manifest.json").read_text())
    assert manifest["queries_sha256"] == sha256_bytes(fresh)
    assert manifest["corpus_chunks_sha256"] == corpus.sha256


def test_dataset_passes_all_integrity_and_leakage_checks(corpus) -> None:  # type: ignore[no-untyped-def]
    queries = load_materialised(ROOT / "data/eval/queries.jsonl")
    report = check_dataset(queries, corpus)
    assert report.ok, report.errors
    assert len(queries) >= 150
    assert all(report.stats["by_split"][s] >= 25 for s in ("train", "val", "test"))  # type: ignore[index]


def test_label_provenance_is_never_overstated() -> None:
    queries = load_materialised(ROOT / "data/eval/queries.jsonl")
    for q in queries:
        if q.label_source == "claude_authored":
            assert q.verification == "llm_authored_unreviewed"
        else:
            assert (q.label_source, q.verification) == ("official_faq_heading", "official_source")
            assert q.pooled_added == []  # the official slice has no LLM-added judgments
    manifest = json.loads((ROOT / "data/manifests/eval_dataset_manifest.json").read_text())
    assert manifest["human_reviewed_queries"] == 0
    assert Counter(q.slice for q in queries) == manifest["by_slice"]


def test_bm25_on_the_real_corpus_finds_obvious_targets(corpus) -> None:  # type: ignore[no-untyped-def]
    bm25 = BM25Retriever(corpus.chunks)
    hits = bm25.search("UnboundLocalError referenced before assignment", k=5)
    assert any(h.section_id.endswith("library/exceptions#UnboundLocalError") for h in hits)
    hits = bm25.search("CORSMiddleware allow_origins", k=3, framework="fastapi")
    assert hits[0].section_id.startswith("fastapi/0.162.0/tutorial/cors#")
    assert all(h.url.startswith("https://") and h.framework == "fastapi" for h in hits)
