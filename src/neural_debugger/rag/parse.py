"""Parse and schema-check the model's JSON reply. Never raises on bad model output: returns a reason."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MAX_TEXT = 1500
MAX_CODE = 4000
MAX_ITEMS = 8
_CITE = re.compile(r"E\d{1,3}", re.I)


def _ids(value: object) -> list[str]:
    """Accept ``["E1"]``, ``"E1, E2"``, ``["[E1]"]``... and return upper-case ids (validity is checked later)."""
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in items:
        for m in _CITE.finditer(str(item)):
            cid = m.group(0).upper()
            if cid not in out:
                out.append(cid)
    return out


class _Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    citations: list[str] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _norm(cls, v: object) -> list[str]:
        return _ids(v) if v else []

    @field_validator("text", mode="before")
    @classmethod
    def _clip(cls, v: object) -> str:
        return str(v).strip()[:MAX_TEXT]


class _Code(BaseModel):
    model_config = ConfigDict(extra="ignore")
    language: str = "python"
    code: str = Field(min_length=1, max_length=MAX_CODE)
    explanation: str = ""
    citations: list[str] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _norm(cls, v: object) -> list[str]:
        return _ids(v) if v else []

    @field_validator("code", mode="before")
    @classmethod
    def _clip(cls, v: object) -> str:
        return str(v)[:MAX_CODE]


def _caveat_list(v: object) -> list[str]:
    if not v:
        return []
    items = v if isinstance(v, list) else [v]
    out: list[str] = []
    for item in items[:MAX_ITEMS]:
        text = item.get("text") if isinstance(item, dict) else item
        if isinstance(text, str) and text.strip():
            out.append(text.strip()[:MAX_TEXT])
    return out


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Literal["answer", "insufficient_evidence"] = "answer"
    diagnosis: _Claim | None = None
    steps: list[_Claim] = Field(default_factory=list)
    code_suggestions: list[_Code] = Field(default_factory=list)
    conflicts: list[_Claim] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)

    @field_validator("caveats", "missing_information", mode="before")
    @classmethod
    def _strings(cls, v: object) -> list[str]:
        return _caveat_list(v)

    @field_validator("steps", "code_suggestions", "conflicts", mode="after")
    @classmethod
    def _cap(cls, v: list[object]) -> list[object]:
        return v[:MAX_ITEMS]


@dataclass
class ParseResult:
    answer: ModelAnswer | None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _first_json_object(text: str) -> str | None:
    """The first balanced ``{...}`` in ``text`` (models sometimes wrap JSON in prose or fences)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_model_output(text: str) -> ParseResult:
    blob = _first_json_object(text)
    if blob is None:
        return ParseResult(None, "no JSON object found in the reply")
    try:
        raw = json.loads(blob)
    except json.JSONDecodeError as exc:
        return ParseResult(None, f"invalid JSON ({exc.msg} at position {exc.pos})")
    if not isinstance(raw, dict):
        return ParseResult(None, "the JSON value is not an object")
    notes: list[str] = []
    if isinstance(raw.get("diagnosis"), str):  # tolerate a bare string diagnosis (uncited)
        raw["diagnosis"] = {"text": raw["diagnosis"], "citations": []}
        notes.append("diagnosis was a plain string")
    try:
        return ParseResult(ModelAnswer.model_validate(raw), None, notes)
    except ValidationError as exc:
        first = exc.errors()[0]
        return ParseResult(
            None, f"schema error at {'.'.join(map(str, first['loc']))}: {first['msg']}"
        )
