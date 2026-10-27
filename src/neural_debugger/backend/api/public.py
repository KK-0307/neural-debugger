"""Public, non-secret deployment facts the UI needs (which mode this server runs in, what is sent where)."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["public"])

FREE_TIER_NOTICE = (
    "Your text is sent to Google's Gemini API on the free tier. Google may use free-tier prompts and responses to improve "
    "its products. Do not paste passwords, API keys or private code."
)


class PublicConfig(BaseModel):
    mode: Literal["off", "readonly"]
    llm_provider: str
    max_question_chars: int | None
    rate_per_minute: int | None
    rate_per_day: int | None
    notice: str | None


@router.get("/config", response_model=PublicConfig)
def public_config(request: Request) -> PublicConfig:
    s = request.app.state.settings
    readonly = s.demo_mode == "readonly"
    return PublicConfig(
        mode=s.demo_mode,
        llm_provider=s.llm_provider,
        max_question_chars=s.demo_max_question_chars if readonly else None,
        rate_per_minute=s.demo_rate_per_minute if readonly else None,
        rate_per_day=s.demo_rate_per_day if readonly else None,
        notice=FREE_TIER_NOTICE if s.llm_provider == "gemini" else None,
    )
