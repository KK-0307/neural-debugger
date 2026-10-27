"""Request-ID propagation, access logging, and last-resort 500 handling."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from .errors import error_response

logger = logging.getLogger("neural_debugger.access")

REQUEST_ID_HEADER = "X-Request-ID"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "unhandled_exception",
            extra={"request_id": request_id, "path": request.url.path},
            exc_info=(type(exc), exc, None),
        )
        response = error_response(500, "internal_error", "Internal server error.", request_id)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers[REQUEST_ID_HEADER] = request_id
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,  # path only; query strings can carry user data
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response
