"""Database engine and connectivity check."""

import time
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text


def create_db_engine(database_url: str) -> Engine:
    kwargs: dict[str, object] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        # In-process sqlite (tests / quick local runs) is shared across threads.
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **kwargs)


@dataclass(frozen=True)
class DbCheck:
    ok: bool
    latency_ms: float
    error_type: str | None = None


def check_database(engine: Engine) -> DbCheck:
    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        return DbCheck(False, round((time.perf_counter() - start) * 1000, 2), type(exc).__name__)
    return DbCheck(True, round((time.perf_counter() - start) * 1000, 2))
