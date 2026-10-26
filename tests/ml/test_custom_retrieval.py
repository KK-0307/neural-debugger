"""The from-scratch model as a retrieval system: encoder adapter, config merging, stale-embedding rebuild."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from neural_debugger.ml.retrieval.custom_encoder import CustomEncoder  # noqa: E402
from neural_debugger.ml.retrieval.systems import (  # noqa: E402
    SystemFactory,
    load_neural_config,
)
from neural_debugger.ml.training.train import run_training  # noqa: E402
from tests.ml.test_training import tiny_config, tiny_corpus  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("run") / "tiny"
    run_training(tiny_config("tiny", epochs=3), tiny_corpus(), [], run, log=lambda *_: None)
    return run


def test_encoder_outputs_unit_vectors_and_is_deterministic(trained: Path) -> None:
    enc = CustomEncoder(trained, max_query_len=24, max_doc_len=64)
    texts = ["zw3x1 value controls", "Topic 4\nTiny\nThe zw4x2 value"]
    a, b = enc.encode_queries(texts), enc.encode_queries(texts)
    assert a.shape == (2, enc.dim) and a.dtype == np.float32 and np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert enc.encode_documents([]).shape == (0, enc.dim)
    fp = enc.fingerprint()
    assert fp["kind"] == "custom-transformer" and len(fp["checkpoint_sha256"]) == 64
    # batching and ordering must not change the vectors
    one_by_one = np.vstack([enc.encode_queries([t], batch_size=1) for t in texts])
    assert np.allclose(a, one_by_one, atol=1e-5)


def test_encoder_rejects_a_mismatched_tokenizer(trained: Path, tmp_path: Path) -> None:
    import shutil

    from neural_debugger.ml.training.tokenizer import TextTokenizer

    shutil.copytree(trained, tmp_path / "m")
    TextTokenizer.train(["something else entirely"] * 5, vocab_size=60, min_frequency=1).save(
        tmp_path / "m" / "tokenizer.json"
    )
    with pytest.raises(ValueError, match="tokenizer"):
        CustomEncoder(tmp_path / "m")


def test_custom_model_builds_as_a_dense_and_hybrid_system(trained: Path, tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("systems:\n  bm25: {kind: bm25}\n")
    extra = tmp_path / "m4.yaml"
    extra.write_text(
        f"custom_models:\n  c: {{model_dir: {trained}, max_query_len: 24, max_doc_len: 64}}\n"
        "systems:\n  dense-c: {kind: dense, model: c}\n"
        "  hybrid-c: {kind: hybrid, components: [bm25, dense-c], weights: [1.0, 1.0], rrf_k: 10}\n"
    )
    cfg = load_neural_config(base, extra)
    corpus = tiny_corpus()
    factory = SystemFactory(corpus, cfg, embedding_dir=tmp_path / "emb")
    hits = factory.build("dense-c").search("zw7x1 value controls zw7x2", k=5)
    assert hits and hits[0].section_id.startswith(corpus.chunks[0].doc_id)
    assert (tmp_path / "emb" / "c" / "embeddings.npy").exists()
    assert factory.build("hybrid-c").search("zw7x1 value controls", k=5)
    # a second factory reuses the stored embeddings (no re-embedding)
    again = SystemFactory(corpus, cfg, embedding_dir=tmp_path / "emb")
    again.build("dense-c")
    assert (
        again.info.embedding["c"]["embedding_seconds"]
        == factory.info.embedding["c"]["embedding_seconds"]
    )


def test_config_merge_rejects_redefinition(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("systems:\n  bm25: {kind: bm25}\n")
    b.write_text("systems:\n  bm25: {kind: bm25}\n")
    with pytest.raises(ValueError, match="redefines"):
        load_neural_config(a, b)
