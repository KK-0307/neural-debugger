"""Neural-retrieval plumbing, tested with deterministic real encoders/scorers (no model download)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_debugger.data.corpus import Corpus  # noqa: E402
from neural_debugger.ml.retrieval.bm25 import BM25Retriever  # noqa: E402
from neural_debugger.ml.retrieval.dense import DenseRetriever  # noqa: E402
from neural_debugger.ml.retrieval.embedding_store import (  # noqa: E402
    StaleEmbeddingsError,
    embed_chunks,
    ensure_embeddings,
    load_embeddings,
)
from neural_debugger.ml.retrieval.encoders import HashingEncoder  # noqa: E402
from neural_debugger.ml.retrieval.framework import FrameworkDetector, FrameworkRouter  # noqa: E402
from neural_debugger.ml.retrieval.fusion import (  # noqa: E402
    HybridRetriever,
    fuse_hits,
    rrf_scores,
)
from neural_debugger.ml.retrieval.reranker import RerankedRetriever  # noqa: E402
from neural_debugger.ml.retrieval.systems import (  # noqa: E402
    NeuralConfig,
    SystemFactory,
    SystemSpec,
)
from neural_debugger.ml.retrieval.vector_index import (  # noqa: E402
    VectorIndex,
    faiss_available,
    faiss_is_safe,
)
from tests.helpers import make_chunk  # noqa: E402

CHUNKS = [
    make_chunk(
        "keyerror", "A KeyError is raised when a dictionary key is not found", framework="python"
    ),
    make_chunk(
        "indexerror",
        "An IndexError is raised when a list index is out of range",
        framework="python",
    ),
    make_chunk(
        "device", "Tensors must be on the same cuda device before an operation", framework="pytorch"
    ),
    make_chunk(
        "backward",
        "Call backward on a scalar loss tensor to compute gradients",
        framework="pytorch",
    ),
    make_chunk(
        "cors",
        "Use CORSMiddleware to allow origins in your FastAPI application",
        framework="fastapi",
    ),
    make_chunk(
        "depends",
        "Declare a dependency with Depends in a FastAPI path operation",
        framework="fastapi",
    ),
]


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(chunks=CHUNKS, documents=[], sha256="corpus-sha-1")


# ----------------------------------------------------------------------------- vector index
FAISS_CHECK = """
import json, numpy as np
from neural_debugger.ml.retrieval.vector_index import VectorIndex
rng = np.random.RandomState(0)
v = rng.randn(200, 32).astype(np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True)
groups = ["a", "b", "c"] * 66 + ["a", "b"]
f, n = VectorIndex(v, groups, backend="faiss"), VectorIndex(v, groups, backend="numpy")
bad = 0
for _ in range(10):
    q = rng.randn(32).astype(np.float32); q /= np.linalg.norm(q)
    for g in (None, "a", "c"):
        sf, idf = f.search(q, 10, g); sn, idn = n.search(q, 10, g)
        bad += int(list(idf) != list(idn) or not np.allclose(sf, sn, atol=1e-5))
        if g: bad += int({groups[i] for i in idf} != {g})
print(json.dumps({"backend": f.backend, "bad": bad}))
"""


@pytest.mark.skipif(not faiss_available(), reason="faiss not installed")
def test_faiss_backend_agrees_with_numpy_including_group_filters() -> None:
    # Run in a subprocess that never imports torch: on macOS faiss + torch in one process abort
    # (duplicate libomp), which is exactly why backend="auto" avoids FAISS when torch is loaded.
    out = subprocess.run(
        [sys.executable, "-c", FAISS_CHECK], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"backend": "faiss", "bad": 0}


def test_numpy_backend_is_exact_sorted_and_deterministic() -> None:
    rng = np.random.RandomState(1)
    v = rng.randn(100, 16).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    groups = ["x", "y"] * 50
    idx = VectorIndex(v, groups, backend="numpy")
    q = v[3]
    scores, ids = idx.search(q, 5)
    brute = np.argsort(-(v @ q), kind="stable")[:5]
    assert list(ids) == list(brute) and ids[0] == 3 and np.all(np.diff(scores) <= 1e-7)
    assert {groups[i] for i in idx.search(q, 20, "y")[1]} == {"y"}
    dup = np.tile(v[:1], (4, 1))
    tie_ids = VectorIndex(dup, ["g"] * 4, backend="numpy").search(v[0], 4)[1]
    assert list(tie_ids) == [0, 1, 2, 3]  # ties -> lower row id


def test_auto_backend_never_picks_faiss_when_it_is_unsafe() -> None:
    idx = VectorIndex(np.eye(4, dtype=np.float32), ["a"] * 4)
    assert idx.backend == ("faiss" if faiss_is_safe() else "numpy")
    if sys.platform == "darwin" and "torch" in sys.modules:
        assert idx.backend == "numpy"


def test_index_handles_k_larger_than_group_and_validates_inputs() -> None:
    v = np.eye(4, dtype=np.float32)
    idx = VectorIndex(v, ["x", "x", "y", "y"], backend="numpy")
    assert len(idx.search(v[0], 100, "x")[1]) == 2
    with pytest.raises(KeyError):
        idx.search(v[0], 3, "nope")
    with pytest.raises(ValueError):
        VectorIndex(v, ["x"])


# ---------------------------------------------------------------------------- embedding store
class CountingEncoder:
    """Real HashingEncoder that counts how many texts were encoded (to prove cache reuse)."""

    def __init__(self, dim: int = 64) -> None:
        self.inner = HashingEncoder(dim=dim)
        self.name, self.dim, self.calls = "counting", dim, 0

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        self.calls += len(texts)
        return self.inner.encode_documents(texts)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        return self.inner.encode_queries(texts)

    def fingerprint(self) -> dict[str, Any]:
        return self.inner.fingerprint()


def test_embed_chunks_preserves_corpus_order_and_is_deterministic() -> None:
    enc = HashingEncoder(dim=64)
    out = embed_chunks(enc, CHUNKS, batch_size=2)
    expected = enc.encode_documents([c.index_text() for c in CHUNKS])
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out, embed_chunks(enc, CHUNKS, batch_size=5))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embeddings_are_persisted_reused_and_invalidated_on_change(tmp_path: Path) -> None:
    enc = CountingEncoder()
    first = ensure_embeddings(tmp_path / "m", enc, CHUNKS, "sha-1")
    assert enc.calls == len(CHUNKS) and first.meta["embedding_seconds"] >= 0
    again = ensure_embeddings(tmp_path / "m", enc, CHUNKS, "sha-1")
    assert enc.calls == len(CHUNKS)  # loaded from disk, not re-encoded
    np.testing.assert_array_equal(first.vectors, again.vectors)
    # different corpus hash -> stale -> rebuilt (never serve vectors for another corpus)
    ensure_embeddings(tmp_path / "m", enc, CHUNKS, "sha-2")
    assert enc.calls == 2 * len(CHUNKS)
    # different encoder fingerprint (dim) -> stale
    other = CountingEncoder(dim=32)
    with pytest.raises(StaleEmbeddingsError, match="encoder.dim"):
        load_embeddings(tmp_path / "m", other, "sha-2", len(CHUNKS))
    ensure_embeddings(tmp_path / "m", other, CHUNKS, "sha-2")
    assert other.calls == len(CHUNKS)
    # forced rebuild
    ensure_embeddings(tmp_path / "m", other, CHUNKS, "sha-2", rebuild=True)
    assert other.calls == 2 * len(CHUNKS)


# -------------------------------------------------------------------------------- dense retrieval
@pytest.fixture
def dense(corpus: Corpus) -> DenseRetriever:
    enc = HashingEncoder(dim=512)
    return DenseRetriever(corpus.chunks, enc, embed_chunks(enc, corpus.chunks))


def test_dense_retriever_finds_the_matching_section_and_carries_metadata(
    dense: DenseRetriever,
) -> None:
    hits = dense.search("backward on a scalar loss tensor gradients", k=3)
    assert hits[0].section_id.endswith("#backward")
    h = hits[0]
    assert (h.framework, h.rank) == ("pytorch", 1) and h.url.endswith("#backward") and h.snippet
    assert [x.rank for x in hits] == [1, 2, 3] and hits[0].score >= hits[1].score >= hits[2].score


def test_dense_framework_filter_and_section_collapse() -> None:
    chunks = [
        make_chunk("s", "tensor device cuda", framework="pytorch", ordinal=0),
        make_chunk("s", "tensor device cuda again", framework="pytorch", ordinal=1),
        make_chunk("t", "tensor device cuda", framework="python"),
    ]
    enc = HashingEncoder(dim=128)
    d = DenseRetriever(chunks, enc, embed_chunks(enc, chunks))
    hits = d.search("tensor device cuda", k=10)
    assert len([h for h in hits if h.section_id.endswith("#s")]) == 1
    assert {h.framework for h in d.search("tensor device cuda", k=10, framework="python")} == {
        "python"
    }


# ------------------------------------------------------------------------------------- fusion
def test_rrf_scores_match_hand_computation() -> None:
    s = rrf_scores([["a", "b", "c"], ["c", "a"]], k=60)
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["b"] == pytest.approx(1 / 62)
    assert s["c"] == pytest.approx(1 / 63 + 1 / 61)
    w = rrf_scores([["a"], ["a"]], weights=[2.0, 0.5], k=10)
    assert w["a"] == pytest.approx(2.0 / 11 + 0.5 / 11)
    with pytest.raises(ValueError):
        rrf_scores([["a"]], weights=[1, 2])


def test_fusion_promotes_agreement_and_keeps_single_list_items(
    corpus: Corpus, dense: DenseRetriever
) -> None:
    bm25 = BM25Retriever(corpus.chunks)
    q = "KeyError dictionary key not found"
    lists = [bm25.search(q, 10), dense.search(q, 10)]
    fused = fuse_hits(lists, k_rrf=60)
    assert fused[0].section_id.endswith("#keyerror")
    in_any = {h.section_id for hl in lists for h in hl}
    assert {h.section_id for h in fused} == in_any  # nothing dropped, nothing invented
    assert [h.rank for h in fused] == list(range(1, len(fused) + 1))
    assert all(a.score >= b.score for a, b in zip(fused, fused[1:], strict=False))


def test_hybrid_retriever_is_deterministic_and_weights_matter(
    corpus: Corpus, dense: DenseRetriever
) -> None:
    bm25 = BM25Retriever(corpus.chunks)
    q = "device"
    h1 = HybridRetriever([bm25, dense], [1.0, 1.0]).search(q, 6)
    assert [h.chunk_id for h in h1] == [
        h.chunk_id for h in HybridRetriever([bm25, dense]).search(q, 6)
    ]
    only_dense = HybridRetriever([bm25, dense], [0.0, 1.0]).search(q, 6)
    assert [h.section_id for h in only_dense][:3] == [h.section_id for h in dense.search(q, 3)]


# ------------------------------------------------------------------------------------ framework
def test_detector_learns_frameworks_from_the_corpus_only(corpus: Corpus) -> None:
    det = FrameworkDetector(corpus.chunks)
    assert det.predict("CORSMiddleware origins FastAPI")[0] == "fastapi"
    assert det.predict("cuda tensor gradients backward")[0] == "pytorch"
    assert det.predict("dictionary key list index")[0] == "python"
    p = det.predict_proba("cuda tensor")
    assert sum(p.values()) == pytest.approx(1.0)
    assert det.predict_proba("zzz unknown words")  # no crash on out-of-vocabulary input


def test_router_hard_soft_and_explicit_framework(corpus: Corpus, dense: DenseRetriever) -> None:
    det = FrameworkDetector(corpus.chunks)
    base = HybridRetriever([BM25Retriever(corpus.chunks), dense])
    q = "cuda tensor device gradients"
    hard = FrameworkRouter(base, det, mode="hard", threshold=0.6)
    assert {h.framework for h in hard.search(q, 6)} == {"pytorch"}
    unsure = FrameworkRouter(base, det, mode="hard", threshold=1.01)  # never confident -> no filter
    assert len({h.framework for h in unsure.search(q, 6)}) > 1
    explicit = hard.search(q, 6, framework="python")
    assert {h.framework for h in explicit} == {"python"}  # user choice always wins
    soft = FrameworkRouter(base, det, mode="soft", beta=5.0)
    assert soft.search(q, 3)[0].framework == "pytorch"
    with pytest.raises(ValueError):
        FrameworkRouter(base, det, mode="x")


# ------------------------------------------------------------------------------------ reranker
class OverlapScorer:
    """A real pair scorer: fraction of query words present in the passage (deterministic)."""

    name = "overlap"

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int = 16) -> list[float]:
        out = []
        for q, p in pairs:
            qs, ps = set(q.lower().split()), set(p.lower().split())
            out.append(len(qs & ps) / max(1, len(qs)))
        return out

    def fingerprint(self) -> dict[str, Any]:
        return {"kind": "overlap"}


def test_reranker_reorders_head_and_preserves_tail(corpus: Corpus, dense: DenseRetriever) -> None:
    by_id = {c.chunk_id: c for c in corpus.chunks}
    base = dense
    q = "dependency Depends FastAPI path operation"
    base_hits = base.search(q, 6)
    rr = RerankedRetriever(base, OverlapScorer(), by_id, top_n=3, mode="replace", depth=6)
    out = rr.search(q, 6)
    assert [h.section_id for h in out[3:]] == [
        h.section_id for h in base_hits[3:]
    ]  # tail untouched
    assert {h.section_id for h in out[:3]} == {h.section_id for h in base_hits[:3]}
    assert out[0].section_id.endswith("#depends")
    assert [h.rank for h in out] == [1, 2, 3, 4, 5, 6]
    mixed = RerankedRetriever(base, OverlapScorer(), by_id, top_n=3, mode="rrf", depth=6).search(
        q, 6
    )
    assert {h.section_id for h in mixed} == {h.section_id for h in out}
    with pytest.raises(ValueError):
        RerankedRetriever(base, OverlapScorer(), by_id, mode="bogus")


# ------------------------------------------------------------------------------ system factory
def test_system_factory_builds_every_system_kind_from_config(
    tmp_path: Path, corpus: Corpus
) -> None:
    cfg = NeuralConfig.model_validate(
        {
            "embedding_models": {"hash": {"hf_id": "n/a"}},
            "rerankers": {"overlap": {"hf_id": "n/a"}},
            "systems": {
                "bm25": {"kind": "bm25"},
                "dense": {"kind": "dense", "model": "hash"},
                "hybrid": {"kind": "hybrid", "components": ["bm25", "dense"], "rrf_k": 30},
                "rerank": {
                    "kind": "rerank",
                    "base": "hybrid",
                    "reranker": "overlap",
                    "top_n": 4,
                    "mode": "rrf",
                },
                "auto": {"kind": "router", "base": "rerank", "mode": "hard", "threshold": 0.6},
            },
        }
    )
    f = SystemFactory(
        corpus,
        cfg,
        embedding_dir=tmp_path,
        encoder_override={"hash": HashingEncoder(dim=256)},
        scorer_override={"overlap": OverlapScorer()},
    )
    for name in cfg.systems:
        hits = f.build(name).search("cuda tensor gradients", 3)
        assert hits and hits[0].url.startswith("https://")
    assert f.build("hybrid") is f.build("hybrid")  # shared components are built once
    assert (tmp_path / "hash" / "meta.json").exists() and f.info.embedding["hash"][
        "n_chunks"
    ] == len(CHUNKS)
    with pytest.raises(ValueError):
        SystemFactory(corpus, cfg, extra_systems={"x": SystemSpec(kind="nope")}).build("x")


def test_soft_router_handles_score_ties_deterministically(
    corpus: Corpus, dense: DenseRetriever
) -> None:
    """Regression: sorting (score, SearchHit) tuples raised TypeError when two scores tied."""
    det = FrameworkDetector(corpus.chunks)
    identical = [
        make_chunk(f"s{i}", "same text for every chunk", framework="python") for i in range(4)
    ]
    enc = HashingEncoder(dim=64)
    d = DenseRetriever(identical, enc, embed_chunks(enc, identical))
    router = FrameworkRouter(d, det, mode="soft", beta=0.0)
    a = [h.section_id for h in router.search("same text", 4)]
    assert a == [h.section_id for h in router.search("same text", 4)] and len(a) == 4
