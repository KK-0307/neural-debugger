"""Real components: BM25 + the actual corpus (always, if the corpus exists), the pretrained hybrid when its
model is cached, and a live LLM only when a key is present (otherwise skipped, and reported as unverified)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")
from neural_debugger.data.corpus import load_corpus  # noqa: E402
from neural_debugger.ml.retrieval.bm25 import BM25Config, BM25Retriever  # noqa: E402
from neural_debugger.rag import RAGPipeline  # noqa: E402
from neural_debugger.rag.providers.testing import ExtractiveProvider  # noqa: E402
from neural_debugger.rag.retrieval import CorpusSectionRetriever  # noqa: E402
from neural_debugger.rag.sufficiency import TermStats  # noqa: E402

CORPUS = Path("data/corpus")
needs_corpus = pytest.mark.skipif(not (CORPUS / "chunks.jsonl").exists(), reason="corpus not built")


@pytest.fixture(scope="module")
def corpus():  # type: ignore[no-untyped-def]
    return load_corpus(CORPUS)


@needs_corpus
def test_bm25_rag_cites_only_real_corpus_sections_and_abstains_out_of_scope(corpus) -> None:  # type: ignore[no-untyped-def]
    pipe = RAGPipeline(
        CorpusSectionRetriever(BM25Retriever(corpus.chunks, BM25Config()), corpus),
        ExtractiveProvider(),
        stats=TermStats.from_texts(c.index_text() for c in corpus.chunks),
    )
    ans = pipe.answer("how do I make a query parameter optional with a default value in FastAPI")
    assert ans.status in ("answered", "partial") and ans.citations
    assert all(
        c.section_id in corpus.section_ids and c.url == corpus.section_url(c.section_id)
        for c in ans.citations
    )
    assert pipe.answer("TensorFlow keras model.fit shape mismatch error").status == "abstained"


@needs_corpus
def test_production_hybrid_backbone_end_to_end_when_model_is_cached(corpus) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("sentence_transformers")
    if not Path("data/embeddings/gte-small/embeddings.npy").exists():
        pytest.skip("embeddings not available")
    from neural_debugger.backend.config import Settings
    from neural_debugger.backend.services.rag import build_pipeline

    try:
        pipe = build_pipeline(
            Settings(environment="test", llm_provider="extractive", retriever_device="cpu")
        )
    except Exception as exc:  # model not in the local HF cache and no network
        pytest.skip(f"hybrid model unavailable: {type(exc).__name__}")
    ans = pipe.answer(
        "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"
    )
    assert ans.status in ("answered", "partial") and ans.citations
    assert {c.section_id for c in ans.citations} <= corpus.section_ids


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY: live generation is UNVERIFIED",
)
@needs_corpus
def test_live_anthropic_generation_end_to_end(corpus) -> None:  # type: ignore[no-untyped-def]
    """Costs a small amount of real money; runs only when the key is deliberately present."""
    from neural_debugger.rag.providers.http import AnthropicProvider

    pipe = RAGPipeline(
        CorpusSectionRetriever(BM25Retriever(corpus.chunks, BM25Config()), corpus),
        AnthropicProvider(os.environ.get("ND_LLM_MODEL", "claude-haiku-5-5")),
        stats=TermStats.from_texts(c.index_text() for c in corpus.chunks),
    )
    ans = pipe.answer("How do I make a query parameter optional with a default value in FastAPI?")
    assert (
        ans.status in ("answered", "partial")
        and ans.citations
        and ans.validation.invalid_citations == []
    )
