"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neural_debugger.rag import RAGPipeline

from . import __version__
from .api import debug, feedback, health
from .config import Settings, get_settings
from .db import create_db_engine
from .errors import ErrorResponse, register_exception_handlers
from .logging_config import configure_logging
from .middleware import request_context_middleware
from .services.demo_guard import DemoLimiter
from .services.feedback import FeedbackStore
from .services.rag import PipelineHolder


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = create_db_engine(settings.database_url)
        if settings.rag_warm_on_startup:
            app.state.rag.get()
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
    return app


app = create_app()
