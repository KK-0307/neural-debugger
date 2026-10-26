"""Build retrieval systems from ``configs/neural.yaml`` (named, reproducible configurations)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from neural_debugger.data.corpus import Corpus

from .bm25 import BM25Config, BM25Retriever
from .dense import DenseRetriever
from .embedding_store import EmbeddingSet, ensure_embeddings
from .encoders import Encoder, SentenceTransformerEncoder
from .framework import FrameworkDetector, FrameworkRouter
from .fusion import HybridRetriever
from .reranker import CrossEncoderScorer, PairScorer, RerankedRetriever
from .types import Retriever


class EmbeddingModelSpec(BaseModel):
    hf_id: str
    revision: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq_length: int | None = None
    license: str = ""
    size_mb: float = 0


class CustomModelSpec(BaseModel):
    """A from-scratch transformer checkpoint produced by ``ml/training`` (see docs/ml-model.md)."""

    model_dir: str
    checkpoint: str = "best.pt"
    max_query_len: int = 64
    max_doc_len: int = 256
    license: str = "MIT (this repository)"


class RerankerSpec(BaseModel):
    hf_id: str
    revision: str | None = None
    max_length: int = 512
    license: str = ""
    size_mb: float = 0


class SystemSpec(BaseModel):
    kind: str
    model: str | None = None
    components: list[str] = Field(default_factory=list)
    weights: list[float] | None = None
    rrf_k: int = 60
    depth: int = 100
    base: str | None = None
    reranker: str | None = None
    top_n: int = 30
    mode: str | None = None
    threshold: float = 0.9
    beta: float = 1.0


class NeuralConfig(BaseModel):
    embedding_dir: str = "data/embeddings"
    embedding_models: dict[str, EmbeddingModelSpec] = Field(default_factory=dict)
    custom_models: dict[str, CustomModelSpec] = Field(default_factory=dict)
    rerankers: dict[str, RerankerSpec] = Field(default_factory=dict)
    systems: dict[str, SystemSpec]


def load_neural_config(
    path: str | Path = "configs/neural.yaml", *extra: str | Path
) -> NeuralConfig:
    """Load ``path`` and merge any ``extra`` files on top (later files add keys to each section).

    ``configs/neural.yaml`` is hashed by the M3 freeze manifest, so later milestones add their models
    and systems in separate files (``configs/m4.yaml``) instead of editing it.
    """
    merged: dict[str, Any] = {}
    for f in (path, *extra):
        for section, body in (yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}).items():
            if isinstance(body, dict):
                clash = set(merged.get(section, {})) & set(body)
                if clash:
                    raise ValueError(f"{f}: {section} redefines {sorted(clash)}")
                merged.setdefault(section, {}).update(body)
            else:
                merged[section] = body
    return NeuralConfig.model_validate(merged)


@dataclass
class BuildInfo:
    """Costs observed while building one system (for the evaluation report)."""

    seconds: dict[str, float] = field(default_factory=dict)
    embedding: dict[str, dict[str, Any]] = field(default_factory=dict)


class SystemFactory:
    """Builds and caches systems/encoders so shared components (BM25, encoders) are built once."""

    def __init__(
        self,
        corpus: Corpus,
        config: NeuralConfig,
        device: str = "auto",
        extra_systems: dict[str, SystemSpec] | None = None,
        encoder_override: dict[str, Encoder] | None = None,
        embedding_dir: Path | None = None,
        scorer_override: dict[str, PairScorer] | None = None,
    ) -> None:
        self.corpus, self.config, self.device = corpus, config, device
        self.systems = {**config.systems, **(extra_systems or {})}
        self.embedding_dir = embedding_dir or Path(config.embedding_dir)
        self._encoder_override = encoder_override or {}
        self._scorer_override = scorer_override or {}
        self._cache: dict[str, Retriever] = {}
        self._detector: FrameworkDetector | None = None
        self._scorers: dict[str, PairScorer] = {}
        self.info = BuildInfo()
        self._chunk_by_id = {c.chunk_id: c for c in corpus.chunks}

    def encoder(self, model: str) -> Encoder:
        if model in self._encoder_override:
            return self._encoder_override[model]
        if model in self.config.custom_models:
            from .custom_encoder import CustomEncoder

            c = self.config.custom_models[model]
            return CustomEncoder(
                c.model_dir,
                c.checkpoint,
                c.max_query_len,
                c.max_doc_len,
                device="cpu" if self.device == "auto" else self.device,
                name=model,
            )
        spec = self.config.embedding_models[model]
        return SentenceTransformerEncoder(
            hf_id=spec.hf_id,
            revision=spec.revision,
            query_prefix=spec.query_prefix,
            doc_prefix=spec.doc_prefix,
            max_seq_length=spec.max_seq_length,
            device=self.device,
            name=model,
        )

    def embeddings(self, model: str, encoder: Encoder) -> EmbeddingSet:
        started = time.perf_counter()
        emb = ensure_embeddings(
            self.embedding_dir / model, encoder, self.corpus.chunks, self.corpus.sha256
        )
        self.info.seconds[f"embeddings_load_or_build:{model}"] = time.perf_counter() - started
        self.info.embedding[model] = emb.meta
        return emb

    def scorer(self, name: str) -> PairScorer:
        if name not in self._scorers:
            if name in self._scorer_override:
                self._scorers[name] = self._scorer_override[name]
            else:
                spec = self.config.rerankers[name]
                self._scorers[name] = CrossEncoderScorer(
                    hf_id=spec.hf_id,
                    revision=spec.revision,
                    max_length=spec.max_length,
                    device=self.device,
                    name=name,
                )
        return self._scorers[name]

    def detector(self) -> FrameworkDetector:
        if self._detector is None:
            self._detector = FrameworkDetector(self.corpus.chunks)
        return self._detector

    def build(self, name: str) -> Retriever:
        if name in self._cache:
            return self._cache[name]
        spec = self.systems[name]
        started = time.perf_counter()
        r: Retriever
        if spec.kind == "bm25":
            r = BM25Retriever(self.corpus.chunks, BM25Config())
        elif spec.kind == "dense":
            assert spec.model
            enc = self.encoder(spec.model)
            emb = self.embeddings(spec.model, enc)
            r = DenseRetriever(self.corpus.chunks, enc, emb.vectors, name=name)
        elif spec.kind == "hybrid":
            r = HybridRetriever(
                [self.build(c) for c in spec.components], spec.weights, spec.rrf_k, spec.depth, name
            )
        elif spec.kind == "rerank":
            assert spec.base and spec.reranker
            r = RerankedRetriever(
                self.build(spec.base),
                self.scorer(spec.reranker),
                self._chunk_by_id,
                spec.top_n,
                spec.mode or "replace",
                spec.rrf_k,
                spec.depth,
                name,
            )
        elif spec.kind == "router":
            assert spec.base
            r = FrameworkRouter(
                self.build(spec.base),
                self.detector(),
                spec.mode or "hard",
                spec.threshold,
                spec.beta,
                spec.depth,
                name,
            )
        else:
            raise ValueError(f"unknown system kind {spec.kind!r}")
        self.info.seconds[f"build:{name}"] = time.perf_counter() - started
        self._cache[name] = r
        return r
