"""Construction of the RAG pipeline from settings (lazy: loading models takes seconds)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from neural_debugger.rag import PipelineConfig, RAGPipeline
from neural_debugger.rag.providers.base import LLMProvider, RetryPolicy
from neural_debugger.rag.providers.testing import ExtractiveProvider, UnconfiguredProvider
from neural_debugger.rag.sufficiency import TermStats

from ..config import Settings
from ..errors import AppError

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    kind = settings.llm_provider
    retry = RetryPolicy(max_retries=settings.llm_max_retries)
    if kind == "none":
        return UnconfiguredProvider(
            "no LLM provider is configured: set ND_LLM_PROVIDER=anthropic (and ANTHROPIC_API_KEY) "
            "or openai_compatible"
        )
    if kind == "extractive":
        if settings.environment == "production":
            raise RuntimeError("the extractive test provider is not allowed in production")
        return ExtractiveProvider()
    from neural_debugger.rag.providers.http import AnthropicProvider, OpenAICompatibleProvider

    if kind == "anthropic":
        return AnthropicProvider(
            settings.llm_model,
            api_key_env=settings.llm_api_key_env,
            base_url=settings.llm_base_url or "https://api.anthropic.com",
            retry=retry,
        )
    local = bool(settings.llm_base_url) and any(
        h in settings.llm_base_url for h in ("localhost", "127.0.0.1", "[::1]")
    )
    return OpenAICompatibleProvider(
        settings.llm_model,
        base_url=settings.llm_base_url or "https://api.openai.com/v1",
        api_key_env=None if local else settings.llm_api_key_env,
        retry=retry,
        key_required=not local,
    )


def build_pipeline(settings: Settings) -> RAGPipeline:
    try:
        from neural_debugger.data.corpus import load_corpus
        from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config
        from neural_debugger.rag.retrieval import CorpusSectionRetriever
    except ImportError as exc:  # numpy / sentence-transformers not installed in this image
        raise AppError(
            "retrieval_unavailable",
            "retrieval dependencies are not installed (install the 'neural' and 'data' extras)",
            503,
        ) from exc
    corpus = load_corpus(Path(settings.corpus_dir))
    cfg = load_neural_config(*settings.retriever_configs)
    try:
        retriever = SystemFactory(corpus, cfg, device=settings.retriever_device).build(
            settings.retriever_system
        )
    except (ImportError, KeyError) as exc:
        raise AppError(
            "retrieval_unavailable",
            f"retriever {settings.retriever_system!r} could not be built: {type(exc).__name__}",
            503,
        ) from exc
    stats = TermStats.from_texts(c.index_text() for c in corpus.chunks)
    return RAGPipeline(
        CorpusSectionRetriever(retriever, corpus),
        build_provider(settings),
        PipelineConfig(
            support=settings.rag_support_check, generation_timeout_s=settings.llm_timeout_seconds
        ),
        stats=stats,
    )


class PipelineHolder:
    """Builds the pipeline on first use (thread-safe) or serves an injected one (tests)."""

    def __init__(self, settings: Settings, pipeline: RAGPipeline | None = None) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(settings.rag_max_concurrent)

    def get(self) -> RAGPipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    logger.info("building RAG pipeline (%s)", self._settings.retriever_system)
                    self._pipeline = build_pipeline(self._settings)
        return self._pipeline
