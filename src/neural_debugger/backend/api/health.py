from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from .. import __version__
from ..db import check_database

router = APIRouter(tags=["health"])


class ComponentStatus(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float
    error_type: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    checks: dict[str, ComponentStatus]


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    """Liveness + dependency check. Returns 503 when the database is unreachable."""
    db = check_database(request.app.state.engine)
    if not db.ok:
        response.status_code = 503
    return HealthResponse(
        status="ok" if db.ok else "degraded",
        version=__version__,
        environment=request.app.state.settings.environment,
        checks={
            "database": ComponentStatus(
                status="ok" if db.ok else "error",
                latency_ms=db.latency_ms,
                error_type=db.error_type,
            )
        },
    )
