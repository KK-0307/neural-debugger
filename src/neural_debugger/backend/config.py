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

    # --- RAG (Milestone 5). Credentials are never settings: only the *name* of the env var is. ---
    corpus_dir: str = "data/corpus"
    retriever_system: str = (
        "hybrid-gte-tuned"  # any system in the configs below (e.g. hybrid-custom-tiny)
    )
    retriever_configs: list[str] = Field(
        default_factory=lambda: ["configs/neural.yaml", "configs/m4.yaml"]
    )
    retriever_device: str = "auto"
    llm_provider: Literal["none", "anthropic", "openai_compatible", "extractive"] = "none"
    llm_model: str = "claude-haiku-5-5"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"
    llm_base_url: str = (
        ""  # default: the provider's public endpoint (the app ignores ANTHROPIC_BASE_URL)
    )
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    rag_support_check: Literal["off", "lexical", "llm"] = "lexical"
    rag_max_concurrent: int = 4
    rag_warm_on_startup: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
