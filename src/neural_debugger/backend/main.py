"""FastAPI application factory."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from neural_debugger.rag import RAGPipeline

from . import __version__
from .api import debug, feedback, health, public
from .config import Settings, get_settings
from .db import create_db_engine
from .errors import ErrorResponse, register_exception_handlers
from .logging_config import configure_logging
from .middleware import request_context_middleware
from .services.demo_guard import DemoLimiter
from .services.feedback import FeedbackStore
from .services.rag import PipelineHolder

logger = logging.getLogger(__name__)


def _warm(app: FastAPI) -> None:
    try:
        app.state.rag.get()
    except Exception as exc:  # noqa: BLE001 - logged by type only; the next request retries and reports a proper error
        logger.error("rag warm-up failed: %s", type(exc).__name__)


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = create_db_engine(settings.database_url)
        if settings.rag_warm_on_startup:
            # in the background: the server accepts connections (and /health answers) while the retriever loads, which is what
            # a host that wakes the container on demand needs; a request that arrives meanwhile waits for the same build
            threading.Thread(target=_warm, args=(app,), name="rag-warmup", daemon=True).start()
        yield
        app.state.engine.dispose()

    app = FastAPI(
        title="Neural Debugger",
        version=__version__,
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    app.state.settings = settings
    app.state.rag = PipelineHolder(settings, pipeline)
    app.state.feedback = FeedbackStore(settings.feedback_dir)
    app.state.demo_limiter = DemoLimiter(
        settings.demo_rate_per_minute, settings.demo_rate_per_day, settings.demo_budget_per_day
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    app.middleware("http")(request_context_middleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(debug.router)
    app.include_router(feedback.router)
    app.include_router(public.router)
    static = Path(settings.static_dir) if settings.static_dir else None
    if static is not None and static.is_dir():
        # mounted last: every API route above wins; unknown paths fall through to the exported site (and its 404 page)
        app.mount("/", StaticFiles(directory=static, html=True), name="site")
    return app


app = create_app()
