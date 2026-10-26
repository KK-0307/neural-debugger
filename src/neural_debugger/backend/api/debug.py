"""``POST /api/v1/debug``: answer a debugging question from the indexed documentation, with citations."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from neural_debugger.rag.types import RAGAnswer

from ..errors import ErrorResponse, error_response
from ..services.rag import PipelineHolder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["debug"])


class DebugRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=20000, description="Question, error message or traceback"
    )
    framework: Literal["python", "pytorch", "fastapi"] | None = Field(
        default=None, description="Restrict retrieval to one documentation set"
    )
    include_trace: bool = False


class ClaimOut(BaseModel):
    text: str
    citations: list[str]
    support: str
    support_score: float | None = None


class CodeOut(BaseModel):
    language: str
    code: str = Field(description="Illustrative text only; never executed by the service")
    explanation: ClaimOut


class CitationOut(BaseModel):
    id: str
    title: str
    heading: str
    url: str
    framework: str
    version: str


class SourceOut(CitationOut):
    cited: bool
    flags: list[str]


class ValidationOut(BaseModel):
    total_claims: int
    dropped_claims: int
    unsupported_claims: int
    partially_supported_claims: int
    invalid_citations: list[str]
    stripped_urls: int


class TimingsOut(BaseModel):
    retrieval_ms: float
    generation_ms: float
    validation_ms: float
    total_ms: float


class DebugResponse(BaseModel):
    request_id: str | None
    status: Literal["answered", "partial", "abstained", "error"]
    confidence: Literal["high", "medium", "low", "none"]
    diagnosis: ClaimOut | None
    steps: list[ClaimOut]
    code_suggestions: list[CodeOut]
    conflicts: list[ClaimOut]
    caveats: list[str]
    missing_information: list[str]
    citations: list[CitationOut]
    sources: list[SourceOut]
    abstain_reason: str | None
    validation: ValidationOut
    timings: TimingsOut
    provider: str | None
    model: str | None
    trace: dict[str, Any] | None = None


def _claim(c: Any) -> ClaimOut:
    return ClaimOut(
        text=c.text, citations=list(c.citations), support=c.support, support_score=c.support_score
    )


def to_response(ans: RAGAnswer, request_id: str | None, include_trace: bool) -> DebugResponse:
    cited = {c.id for c in ans.citations}
    v = ans.validation
    return DebugResponse(
        request_id=request_id,
        status=ans.status,
        confidence=ans.confidence,
        diagnosis=_claim(ans.diagnosis) if ans.diagnosis else None,
        steps=[_claim(s) for s in ans.steps],
        code_suggestions=[
            CodeOut(language=c.language, code=c.code, explanation=_claim(c.explanation))
            for c in ans.code_suggestions
        ],
        conflicts=[_claim(c) for c in ans.conflicts],
        caveats=ans.caveats,
        missing_information=ans.missing_information,
        citations=[
            CitationOut(**{k: getattr(c, k) for k in CitationOut.model_fields})
            for c in ans.citations
        ],
        sources=[
            SourceOut(
                id=e.id,
                title=e.title,
                heading=e.heading,
                url=e.url,
                framework=e.framework,
                version=e.version,
                cited=e.id in cited,
                flags=list(e.flags),
            )
            for e in ans.evidence
        ],
        abstain_reason=ans.abstain_reason,
        validation=ValidationOut(
            total_claims=v.total_claims,
            dropped_claims=v.dropped_claims,
            unsupported_claims=v.unsupported_claims,
            partially_supported_claims=v.partially_supported_claims,
            invalid_citations=v.invalid_citations,
            stripped_urls=len(v.stripped_urls),
        ),
        timings=TimingsOut(**vars(ans.timings)),
        provider=ans.provider,
        model=ans.model,
        trace=ans.trace if include_trace else None,
    )


_ERROR_STATUS = {
    "llm_not_configured": (503, "llm_not_configured"),
    "llm_auth": (503, "llm_unavailable"),
    "llm_rate_limit": (503, "llm_unavailable"),
    "llm_timeout": (504, "llm_timeout"),
    "llm_network": (503, "llm_unavailable"),
    "llm_server": (503, "llm_unavailable"),
    "llm_bad_request": (502, "llm_bad_request"),
    "llm_empty": (502, "llm_bad_response"),
    "malformed_model_output": (502, "bad_model_output"),
    "retrieval_failed": (503, "retrieval_unavailable"),
}


@router.post(
    "/debug",
    response_model=DebugResponse,
    responses={
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def debug(body: DebugRequest, request: Request) -> Any:
    holder: PipelineHolder = request.app.state.rag
    request_id = getattr(request.state, "request_id", None)
    if not holder.slots.acquire(blocking=False):
        return error_response(
            429, "busy", "Too many answers are being generated; retry shortly.", request_id
        )
    try:
        pipeline = holder.get()
        ans = pipeline.answer(body.question, body.framework)
    finally:
        holder.slots.release()
    logger.info(
        "debug status=%s confidence=%s reason=%s evidence=%d cited=%d total_ms=%.0f provider=%s",
        ans.status,
        ans.confidence,
        ans.abstain_reason,
        len(ans.evidence),
        len(ans.citations),
        ans.timings.total_ms,
        ans.provider,
    )
    if ans.status == "error":
        status, code = _ERROR_STATUS.get(ans.abstain_reason or "", (502, "rag_error"))
        sources = [{"id": e.id, "title": e.title, "url": e.url} for e in ans.evidence]
        resp: JSONResponse = error_response(
            status,
            code,
            "No answer could be generated. The most relevant documentation is listed in details.",
            request_id,
            details=[{"sources": sources, "reason": ans.abstain_reason}],
        )
        return resp
    return to_response(ans, request_id, body.include_trace)
