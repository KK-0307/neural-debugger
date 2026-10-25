"""Labeled query set: loading, expansion of id prefixes, and validation against the corpus.

Authoring files (``data/eval/queries_*.yaml``) are the human-edited source; ``pooled_judgments.yaml``
adds relevance judgments made by reviewing top-ranked unlabeled sections. Both are merged into the
materialised ``queries.jsonl`` by ``python -m neural_debugger.data.dataset build``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .corpus import Corpus

PREFIXES = {"fa:": "fastapi/0.162.0/", "py:": "python/3.12/", "pt:": "pytorch/2.14/"}
Slice = Literal["error_message", "question", "official_faq"]
Split = Literal["train", "val", "test"]

LABEL_SOURCES = {
    "official_faq": ("official_faq_heading", "official_source"),
    "error_message": ("claude_authored", "llm_authored_unreviewed"),
    "question": ("claude_authored", "llm_authored_unreviewed"),
}


def expand_section_id(short: str) -> str:
    for prefix, full in PREFIXES.items():
        if short.startswith(prefix):
            return full + short[len(prefix) :]
    return short  # already a full id


class QueryRecord(BaseModel):
    id: str
    query: str = Field(min_length=3)
    slice: Slice
    framework: str = ""
    label_source: str = ""
    verification: str = ""
    authored_relevant: list[str]  # sections the query was written for / the FAQ answer
    pooled_added: list[str] = Field(default_factory=list)  # added by pooled review
    split: Split | None = None

    @field_validator("authored_relevant", "pooled_added", mode="before")
    @classmethod
    def _expand(cls, v: list[str]) -> list[str]:
        return [expand_section_id(s) for s in v]

    @property
    def relevant(self) -> list[str]:
        out = list(self.authored_relevant)
        out += [s for s in self.pooled_added if s not in out]
        return out


def load_authored(eval_dir: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for path in sorted(eval_dir.glob("queries_*.yaml")):
        for row in yaml.safe_load(path.read_text(encoding="utf-8")):
            row = dict(row)
            row["authored_relevant"] = row.pop("relevant")
            source, verification = LABEL_SOURCES[row["slice"]]
            row.setdefault("label_source", source)
            row.setdefault("verification", verification)
            records.append(QueryRecord.model_validate(row))
    pooled_path = eval_dir / "pooled_judgments.yaml"
    if pooled_path.exists():
        pooled = yaml.safe_load(pooled_path.read_text(encoding="utf-8")) or {}
        by_id = {r.id: r for r in records}
        for qid, entry in pooled.items():
            by_id[qid].pooled_added = [expand_section_id(s) for s in entry.get("added", [])]
    for r in records:
        frameworks = {s.split("/", 1)[0] for s in r.relevant}
        if len(frameworks) == 1:
            r.framework = frameworks.pop()
    return records


def load_materialised(path: Path) -> list[QueryRecord]:
    return [
        QueryRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate_queries(queries: list[QueryRecord], corpus: Corpus) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    sections = corpus.section_ids
    for q in queries:
        if q.id in seen:
            errors.append(f"{q.id}: duplicate query id")
        seen.add(q.id)
        if not q.authored_relevant:
            errors.append(f"{q.id}: no authored relevant section")
        for s in q.relevant:
            if s not in sections:
                errors.append(f"{q.id}: relevant section not in corpus: {s}")
        if len({s.split("/", 1)[0] for s in q.relevant}) > 1:
            errors.append(f"{q.id}: relevant sections span several frameworks")
        if len(q.relevant) != len(set(q.relevant)):
            errors.append(f"{q.id}: duplicate relevant sections")
    return errors
