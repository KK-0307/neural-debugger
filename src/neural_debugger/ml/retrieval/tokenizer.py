"""Code-aware tokenizer for BM25 (queries and documents use the same function).

Developer text mixes prose with identifiers (``torch.nn.Module``, ``ModuleNotFoundError``,
``cuda:0``, ``__init__``). In ``identifiers`` mode each such compound emits

* the whole lower-cased compound (``torch.nn.module``, ``modulenotfounderror``), so exact
  identifiers match exactly, and
* its sub-words (``torch``, ``nn``, ``module``; ``module``, ``not``, ``found``, ``error``), so a
  query for "module not found" still reaches ``ModuleNotFoundError``.

``plain`` mode (alphanumeric words only) exists as an ablation baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_COMPOUND = re.compile(r"[A-Za-z0-9_]+(?:[.:][A-Za-z0-9_]+)*")
_SUBWORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_PLAIN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "and",
        "or",
        "for",
        "on",
        "at",
        "by",
        "with",
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "from",
        "into",
        "than",
        "then",
        "so",
        "if",
        "but",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
    ]
)


@dataclass(frozen=True)
class TokenizerConfig:
    mode: Literal["plain", "identifiers"] = "identifiers"
    keep_compounds: bool = True  # identifiers mode: also emit the whole compound
    remove_stopwords: bool = True
    stem: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "keep_compounds": self.keep_compounds,
            "remove_stopwords": self.remove_stopwords,
            "stem": self.stem,
        }


def s_stem(token: str) -> str:
    """Harman's S-stemmer: only plural endings (deterministic, no dictionary)."""
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


def _compound_tokens(match: str, cfg: TokenizerConfig) -> list[str]:
    whole = match.lower()
    parts = [
        p.lower()
        for piece in re.split(r"[.:]", match)
        for sub in piece.split("_")
        for p in _SUBWORD.findall(sub)
    ]
    has_alpha = any(not p.isdigit() for p in parts)
    if not has_alpha:  # versions and plain numbers ("3.12") are kept whole
        return [whole]
    parts = [p for p in parts if not (p.isdigit() and has_alpha)]
    if len(parts) <= 1 and "_" not in match and "." not in match and ":" not in match:
        return [whole]
    out = [whole] if cfg.keep_compounds or len(parts) <= 1 else []
    out.extend(parts)
    return out


def tokenize(text: str, cfg: TokenizerConfig = TokenizerConfig()) -> list[str]:
    if cfg.mode == "plain":
        tokens = _PLAIN.findall(text.lower())
    else:
        tokens = []
        for m in _COMPOUND.finditer(text):
            tokens.extend(_compound_tokens(m.group(0).strip("_") or m.group(0), cfg))
    if cfg.remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if cfg.stem:
        tokens = [s_stem(t) for t in tokens]
    return [t for t in tokens if t]
