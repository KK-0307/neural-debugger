"""Application settings, loaded from environment variables (prefix ``ND_``)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ND_", env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    database_url: str = (
        "postgresql+psycopg://neural_debugger:neural_debugger@localhost:5432/neural_debugger"
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    db_check_timeout_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
