"""Shared builders for the RAG tests."""

from __future__ import annotations

import json
from typing import Any

from neural_debugger.rag import PipelineConfig, RAGPipeline
from neural_debugger.rag.providers.testing import ExtractiveProvider, ScriptedProvider
from neural_debugger.rag.types import RetrievedSection

CUDA_TEXT = (
    "Tensors must live on the same device before they can be combined. Use `tensor.to(device)` to move a "
    "tensor and `model.to(device)` to move a module. Mixing a cuda:0 tensor with a cpu tensor raises a "
    "RuntimeError about two devices.\n\n```python\ndevice = torch.device('cuda:0')\nx = x.to(device)\n```"
)
QUERY_TEXT = (
    "You can declare an optional query parameter by giving it a default value of `None`. FastAPI then "
    "treats the parameter as optional and does not require the client to send it."
)


def section(
    n: int,
    text: str = CUDA_TEXT,
    framework: str = "pytorch",
    version: str = "2.14",
    title: str = "CUDA semantics",
    heading: str | None = None,
    doc: str | None = None,
) -> RetrievedSection:
    doc = doc or f"{framework}/{version}/page{n}"
    return RetrievedSection(
        section_id=f"{doc}#sec{n}",
        rank=n,
        score=1.0 / n,
        framework=framework,
        version=version,
        title=title,
        heading=heading or f"{title} > section {n}",
        url=f"https://docs.example.test/{doc}.html#sec{n}",
        text=text,
    )


class FakeRetriever:
    def __init__(self, sections: list[RetrievedSection] | Exception) -> None:
        self.sections = sections
        self.queries: list[str] = []

    def search(self, query: str, k: int, framework: str | None) -> list[RetrievedSection]:
        self.queries.append(query)
        if isinstance(self.sections, Exception):
            raise self.sections
        return [s for s in self.sections if framework in (None, s.framework)][:k]


def answer_json(**kw: Any) -> str:
    base: dict[str, Any] = {
        "status": "answer",
        "diagnosis": {
            "text": "The tensors are on different devices, one on cuda:0 and one on cpu.",
            "citations": ["E1"],
        },
        "steps": [
            {
                "text": "Move the tensor with `tensor.to(device)` so both are on the same device.",
                "citations": ["E1"],
            }
        ],
    }
    base.update(kw)
    return json.dumps(base)


def pipeline(
    sections: list[RetrievedSection] | Exception | None = None,
    replies: list[Any] | None = None,
    **cfg: Any,
) -> tuple[RAGPipeline, ScriptedProvider | ExtractiveProvider, FakeRetriever]:
    retr = FakeRetriever(
        [section(1), section(2, QUERY_TEXT, "fastapi", "0.162.0", "Query parameters")]
        if sections is None
        else sections
    )
    prov: ScriptedProvider | ExtractiveProvider = (
        ScriptedProvider(replies) if replies is not None else ExtractiveProvider()
    )
    return RAGPipeline(retr, prov, PipelineConfig(**cfg)), prov, retr


QUESTION = "RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu"
