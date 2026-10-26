"""Data types shared by the RAG pipeline stages.

Everything that reaches a caller is a plain dataclass so that stages can be tested in isolation and a
whole request can be serialised into a trace. ``Evidence`` objects are the *only* things an answer may
cite: a citation is valid exactly when it names the id of an ``Evidence`` that was shown to the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AnswerStatus = Literal["answered", "partial", "abstained", "error"]
Confidence = Literal["high", "medium", "low", "none"]


@dataclass(frozen=True)
class RetrievedSection:
    """A retrieved documentation section, independent of which retriever produced it."""

    section_id: str
    rank: int
    score: float
    framework: str
    version: str
    title: str
    heading: str
    url: str
    text: str


@dataclass(frozen=True)
class Evidence:
    """Sanitised, budgeted passage that is shown to the model under a stable id (``E1``...)."""

    id: str
    section_id: str
    framework: str
    version: str
    title: str
    heading: str
    url: str
    text: str
    rank: int
    score: float
    flags: tuple[str, ...] = ()
    version_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionInfo:
    original: str
    text: str  # normalised and length-limited; this is what the model sees (as data)
    retrieval_query: str
    error_lines: tuple[str, ...]
    framework_hint: str | None
    versions: tuple[tuple[str, str], ...]  # (framework, version) the user mentioned
    code_blocks: int
    truncated: bool
    injection_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    """One statement of the answer together with the evidence ids that support it."""

    text: str
    citations: tuple[str, ...]
    kind: Literal["diagnosis", "step", "code", "conflict", "caveat"] = "step"
    support: Literal["supported", "partial", "unsupported", "unchecked"] = "unchecked"
    support_score: float | None = None


@dataclass(frozen=True)
class CodeSuggestion:
    language: str
    code: str  # shown, never executed
    explanation: Claim


@dataclass(frozen=True)
class Citation:
    id: str
    section_id: str
    title: str
    heading: str
    url: str
    framework: str
    version: str


@dataclass
class ValidationReport:
    invalid_citations: list[str] = field(default_factory=list)  # ids the model cited but never saw
    dropped_claims: int = 0  # claims removed for having no valid citation
    unsupported_claims: int = 0  # claims removed/flagged by the support check
    partially_supported_claims: int = 0
    total_claims: int = 0
    citations_checked: int = (
        0  # citation ids the model emitted (explicit + inline), before validation
    )
    stripped_urls: list[str] = field(default_factory=list)
    stripped_markup: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class Timings:
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RAGAnswer:
    status: AnswerStatus
    question: str
    diagnosis: Claim | None
    steps: list[Claim]
    code_suggestions: list[CodeSuggestion]
    conflicts: list[Claim]
    caveats: list[str]
    missing_information: list[str]
    citations: list[
        Citation
    ]  # only citations that are used by the final answer, in first-use order
    confidence: Confidence
    abstain_reason: str | None
    evidence: list[Evidence]  # everything the model was shown (for audit)
    validation: ValidationReport
    timings: Timings
    provider: str | None
    model: str | None
    usage: dict[str, int] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cited_section_ids(self) -> list[str]:
        return [c.section_id for c in self.citations]
