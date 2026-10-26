"""Tests that use the real pinned Hugging Face models. Skipped unless they are in the local cache
(`make embed` / any evaluation run downloads them); never run in CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.model
ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs/neural.yaml").read_text())


def _cached(hf_id: str, revision: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(hf_id, revision=revision, local_files_only=True)
        return True
    except Exception:
        return False


EMB = CFG["embedding_models"]["gte-small"]
CE = CFG["rerankers"]["ce-minilm-l12"]


@pytest.fixture(scope="module")
def encoder():  # type: ignore[no-untyped-def]
    if not _cached(EMB["hf_id"], EMB["revision"]):
        pytest.skip("gte-small not in local cache")
    from neural_debugger.ml.retrieval.encoders import SentenceTransformerEncoder

    return SentenceTransformerEncoder(
        hf_id=EMB["hf_id"],
        revision=EMB["revision"],
        max_seq_length=EMB["max_seq_length"],
        device="cpu",
        name="gte-small",
    )


def test_embeddings_are_normalised_deterministic_and_semantic(encoder) -> None:  # type: ignore[no-untyped-def]
    texts = [
        "RuntimeError: Expected all tensors to be on the same device",
        "move the model to the GPU",
        "bake a chocolate cake",
    ]
    a, b = encoder.encode_documents(texts), encoder.encode_documents(texts)
    assert a.shape == (3, 384) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)  # CPU inference is bitwise deterministic
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert a[0] @ a[1] > a[0] @ a[2]  # device error is closer to "GPU" than to cake


def test_stored_embeddings_match_a_fresh_cpu_encode(encoder) -> None:  # type: ignore[no-untyped-def]
    from neural_debugger.data.corpus import load_corpus

    stored = np.load(ROOT / "data/embeddings/gte-small/embeddings.npy")
    chunks = load_corpus(ROOT / "data/corpus").chunks
    idx = list(range(0, len(chunks), 400))[:12]
    fresh = encoder.encode_documents([chunks[i].index_text() for i in idx])
    cos = (fresh * stored[idx]).sum(axis=1)
    assert cos.min() > 0.9999  # MPS-built stored vectors agree with CPU to 4 decimals


def test_end_to_end_dense_search_over_the_real_corpus(encoder) -> None:  # type: ignore[no-untyped-def]
    from neural_debugger.data.corpus import load_corpus
    from neural_debugger.ml.retrieval.dense import DenseRetriever

    corpus = load_corpus(ROOT / "data/corpus")
    dense = DenseRetriever(
        corpus.chunks, encoder, np.load(ROOT / "data/embeddings/gte-small/embeddings.npy")
    )
    hits = dense.search("how to enable CORS in my API", k=5, framework="fastapi")
    assert hits[0].section_id.startswith("fastapi/0.162.0/tutorial/cors#")
    assert {h.framework for h in hits} == {"fastapi"}
    assert all(h.url.startswith("https://fastapi.tiangolo.com/") for h in hits)


def test_cross_encoder_prefers_the_relevant_passage() -> None:
    if not _cached(CE["hf_id"], CE["revision"]):
        pytest.skip("cross-encoder not in local cache")
    from neural_debugger.ml.retrieval.reranker import CrossEncoderScorer

    scorer = CrossEncoderScorer(
        hf_id=CE["hf_id"], revision=CE["revision"], max_length=256, device="cpu"
    )
    q = "how do I save a PyTorch model checkpoint"
    good = "Use torch.save to serialise the model's state_dict to a file, and torch.load to restore it."
    bad = "Python's argparse module parses command line arguments and generates help messages."
    s = scorer.score([(q, good), (q, bad)])
    assert s[0] > s[1]
