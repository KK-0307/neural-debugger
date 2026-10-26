"""Retrieval-augmented generation with validated citations (Milestone 5). See ``docs/rag.md``."""

from .pipeline import PipelineConfig, RAGPipeline
from .types import RAGAnswer

__all__ = ["PipelineConfig", "RAGAnswer", "RAGPipeline"]
