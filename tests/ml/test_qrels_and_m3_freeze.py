"""Incomplete-judgment handling, the frozen Milestone 3 configuration, and the test-access audit trail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from neural_debugger.data.corpus import Corpus, load_corpus
from neural_debugger.data.queries import QueryRecord, load_materialised
from neural_debugger.ml.evaluation import freeze
from neural_debugger.ml.evaluation.qrels import (
    bm25_pool_judged,
    condense,
    load_qrels,
    unjudged_at_k,
)
from tests.helpers import make_chunk

ROOT = Path(__file__).resolve().parents[2]


def test_unjudged_at_k_and_condensed_lists() -> None:
    ranking = ["a", "x", "b", "y", "c"]
    judged = {"a", "b", "c"}
    assert unjudged_at_k(ranking, judged, 4) == 0.5
    assert unjudged_at_k([], judged) == 0.0
    assert condense(ranking, judged) == ["a", "b", "c"]  # order preserved, unjudged dropped


def _corpus_and_query() -> tuple[Corpus, QueryRecord]:
    chunks = [
        make_chunk("keyerror", "A KeyError is raised when a dictionary key is not found"),
        make_chunk("indexerror", "An IndexError is raised when a list index is out of range"),
        make_chunk("other", "Completely unrelated text about asyncio"),
    ]
    q = QueryRecord(
        id="q1",
        query="KeyError dictionary key",
        slice="question",
        framework="python",
        authored_relevant=["python/1.0/page#keyerror"],
    )
    return Corpus(chunks=chunks, documents=[], sha256="x"), q


def test_original_judged_set_is_relevant_plus_bm25_top_k(tmp_path: Path) -> None:
    corpus, q = _corpus_and_query()
    judged = bm25_pool_judged(corpus, [q])["q1"]
    assert "python/1.0/page#keyerror" in judged and "python/1.0/page#other" not in judged
    qrels = load_qrels([q], corpus, "original", eval_dir=tmp_path)
    assert qrels.relevant["q1"] == {"python/1.0/page#keyerror"}


def test_expanded_qrels_merge_dev_and_test_pool_files_and_mark_reviewed(tmp_path: Path) -> None:
    corpus, q = _corpus_and_query()
    (tmp_path / "pooled_judgments_neural.yaml").write_text(
        yaml.safe_dump(
            {
                "q1": {
                    "added": ["python/1.0/page#indexerror"],
                    "reviewed": ["python/1.0/page#indexerror", "python/1.0/page#other"],
                }
            }
        )
    )
    (tmp_path / "pooled_judgments_neural_test.yaml").write_text(
        yaml.safe_dump({"q1": {"added": [], "reviewed": ["python/1.0/page#extra"]}})
    )
    expanded = load_qrels([q], corpus, "expanded", eval_dir=tmp_path)
    assert expanded.relevant["q1"] == {"python/1.0/page#keyerror", "python/1.0/page#indexerror"}
    assert {"python/1.0/page#other", "python/1.0/page#extra"} <= expanded.judged["q1"]
    original = load_qrels([q], corpus, "original", eval_dir=tmp_path)
    assert original.relevant["q1"] == {"python/1.0/page#keyerror"}  # original is never altered


@pytest.mark.skipif(not (ROOT / "data/eval/m3_freeze.json").exists(), reason="no M3 freeze file")
def test_frozen_m3_configuration_is_intact_and_documented() -> None:
    assert freeze.verify_m3(ROOT) == []
    frozen = json.loads((ROOT / "data/eval/m3_freeze.json").read_text())
    assert set(frozen["test_systems"]) == {
        "bm25",
        "dense-gte-small",
        "hybrid-gte-tuned",
        "rerank-l12-n30-rrf",
    }
    assert (
        frozen["embedding_model"]["revision"] and len(frozen["embedding_model"]["revision"]) == 40
    )
    assert frozen["test_systems"]["hybrid-gte-tuned"]["rrf_k"] == 10


@pytest.mark.skipif(
    not (ROOT / "data/eval/results/test_access_log.jsonl").exists(), reason="no log"
)
def test_test_split_access_is_fully_logged() -> None:
    rows = [
        json.loads(x)
        for x in (ROOT / "data/eval/results/test_access_log.jsonl").read_text().splitlines()
    ]
    assert rows[0]["purpose"].startswith("RETROACTIVE")
    m3 = [r for r in rows[1:] if r["purpose"].startswith("system_eval:")]
    # frozen systems scored on test exactly twice each: original, then expanded judgments
    systems = {"bm25", "dense-gte-small", "hybrid-gte-tuned", "rerank-l12-n30-rrf"}
    for variant in ("original", "expanded"):
        assert {r["purpose"].split(":")[1] for r in m3 if r["purpose"].endswith(variant)} == systems
    assert len(m3) == 8


@pytest.mark.skipif(
    not (ROOT / "data/embeddings/gte-small/meta.json").exists(), reason="no stored embeddings"
)
def test_stored_embeddings_metadata_matches_corpus_and_pinned_model() -> None:
    import numpy as np
    import yaml as _yaml

    corpus = load_corpus(ROOT / "data/corpus")
    meta = json.loads((ROOT / "data/embeddings/gte-small/meta.json").read_text())
    cfg = _yaml.safe_load((ROOT / "configs/neural.yaml").read_text())["embedding_models"][
        "gte-small"
    ]
    assert meta["corpus_chunks_sha256"] == corpus.sha256 and meta["n_chunks"] == len(corpus.chunks)
    assert (
        meta["encoder"]["revision"] == cfg["revision"] and meta["encoder"]["hf_id"] == cfg["hf_id"]
    )
    vectors = np.load(ROOT / "data/embeddings/gte-small/embeddings.npy")
    assert (
        vectors.shape == (len(corpus.chunks), meta["encoder"]["dim"])
        and vectors.dtype == np.float32
    )
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert len(load_materialised(ROOT / "data/eval/queries.jsonl")) >= 150
