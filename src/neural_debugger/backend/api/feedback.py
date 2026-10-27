"""``POST /api/v1/feedback``: let a developer say whether an answer helped (Milestone 6)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..errors import ErrorResponse, error_response
from ..services.feedback import FeedbackStorageFullError, FeedbackStore, redact_secrets

router = APIRouter(prefix="/api/v1", tags=["feedback"])

Reason = Literal[
    "wrong_answer",
    "incomplete",
    "wrong_source",
    "unclear",
    "should_have_answered",
    "should_have_abstained",
    "other",
]


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w.\-]+$")
    rating: Literal["helpful", "not_helpful"]
    reasons: list[Reason] = Field(default_factory=list, max_length=7)
    comment: str = Field(default="", max_length=1000)
    answer_status: Literal["answered", "partial", "abstained", "error"] | None = None
    confidence: Literal["high", "medium", "low", "none"] | None = None
    include_question: bool = Field(
        default=False,
        description="Opt-in: store the question text with this feedback. Off by default because it may be private code.",
    )
    question: str | None = Field(default=None, max_length=20000)


class FeedbackResponse(BaseModel):
    id: str
    stored_question: bool


@router.post(
    "/feedback",
    status_code=201,
    response_model=FeedbackResponse,
    responses={503: {"model": ErrorResponse}},
)
def submit_feedback(body: FeedbackRequest, request: Request) -> JSONResponse | FeedbackResponse:
    store: FeedbackStore = request.app.state.feedback
    keep_question = bool(body.include_question and body.question)
    record = {
        "id": uuid.uuid4().hex,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "request_id": body.request_id,
        "rating": body.rating,
        "reasons": sorted(set(body.reasons)),
        "comment": redact_secrets(body.comment.strip()),
        "answer_status": body.answer_status,
        "confidence": body.confidence,
        "question": redact_secrets(body.question or "") if keep_question else None,
    }
    try:
        store.add(record)
    except FeedbackStorageFullError:
        return error_response(
            503,
            "feedback_storage_full",
            "Feedback storage is full on this server; the answer was not affected.",
            getattr(request.state, "request_id", None),
        )
    return FeedbackResponse(id=record["id"], stored_question=keep_question)
