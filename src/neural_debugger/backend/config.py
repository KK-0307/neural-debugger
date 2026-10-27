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
    # extra official-documentation pages merged in memory over the frozen base corpus (Milestone 5.2); [] = base only
    corpus_supplement_dirs: list[str] = Field(default_factory=lambda: ["data/corpus_supplement"])
    # embeddings of base + supplements; the frozen data/embeddings/ of the base corpus alone is never rewritten
    embedding_dir: str = "data/embeddings_rag"
    rag_profile: Literal["m5", "coverage"] = "coverage"  # "m5" = the Milestone 5 evidence settings
    retriever_system: str = (
        "hybrid-gte-tuned"  # any system in the configs below (e.g. hybrid-custom-tiny)
    )
    retriever_configs: list[str] = Field(
        default_factory=lambda: ["configs/neural.yaml", "configs/m4.yaml"]
    )
    retriever_device: str = "auto"
    llm_provider: Literal["none", "anthropic", "openai_compatible", "gemini", "extractive"] = "none"
    llm_model: str = "claude-haiku-5-5"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"
    llm_base_url: str = (
        ""  # default: the provider's public endpoint (the app ignores ANTHROPIC_BASE_URL)
    )
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_reasoning_effort: str = (
        ""  # openai_compatible/gemini: none|low|medium|high (empty = provider default)
    )
    llm_min_interval_seconds: float = (
        0.0  # client-side spacing between requests (free tiers: set >= 6)
    )
    llm_max_requests: int | None = None  # hard per-process cap on requests sent; None = unlimited
    rag_support_check: Literal["off", "lexical", "llm"] = "lexical"
    # semantic grounding: "rules" (numbers, signatures, identifiers, negation; ~0.2 ms, no extra memory) is the
    # deployment default; "nli" adds a 283 MB cross-encoder (~37 ms/claim on CPU). PipelineConfig itself
    # defaults to "off" so the frozen Milestone 5 evaluations are unchanged.
    rag_semantic_check: Literal["off", "rules", "nli"] = "rules"
    # remove claims the semantic check only *disputes* (NLI / negation signals) instead of flagging them;
    # measured to remove some correct claims on real model output, so off by default
    rag_drop_disputed: bool = False
    feedback_dir: str = "var/feedback"  # append-only JSONL of user feedback; git-ignored
    rag_max_concurrent: int = 4
    rag_warm_on_startup: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
