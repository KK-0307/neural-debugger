"""Typed loader for ``configs/sources.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SourceSpec(BaseModel):
    name: str
    framework: str
    version: str
    kind: Literal["sphinx_html", "git_markdown"]
    license: str
    # sphinx_html
    base_url: str | None = None
    pages: list[str] = Field(default_factory=list)
    # git_markdown
    repo: str | None = None
    ref: str | None = None
    commit: str | None = None
    docs_dir: str | None = None
    site_url: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class CorpusConfig(BaseModel):
    corpus_id: str
    user_agent: str
    request_delay_seconds: float = 1.0
    sources: list[SourceSpec]


def load_config(path: str | Path) -> CorpusConfig:
    with open(path, encoding="utf-8") as f:
        return CorpusConfig.model_validate(yaml.safe_load(f))
