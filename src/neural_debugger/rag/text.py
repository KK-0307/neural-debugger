"""Small text utilities used by several stages (tokenising, normalising, truncating)."""

from __future__ import annotations

import re
import unicodedata

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z0-9]+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿‪‫‬‭‮"))

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "while",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "they",
        "them",
        "their",
        "he",
        "she",
        "not",
        "no",
        "do",
        "does",
        "did",
        "done",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "how",
        "why",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "there",
        "here",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "again",
        "about",
        "after",
        "before",
        "between",
        "through",
        "during",
        "each",
        "any",
        "all",
        "some",
        "more",
        "most",
        "other",
        "such",
        "only",
        "own",
        "same",
        "get",
        "gets",
        "got",
        "getting",
        "make",
        "makes",
        "made",
        "use",
        "used",
        "using",
        "like",
        "want",
        "need",
        "trying",
        "tried",
        "try",
        "work",
        "works",
        "working",
        "please",
        "help",
        "thanks",
        "thank",
        "error",
        "errors",
        "exception",
        "exceptions",
        "code",
        "problem",
        "issue",
        "fix",
        "fixed",
    ]
)


def strip_invisible(text: str) -> str:
    """Remove zero-width/bidi characters and non-printing controls (keeps ``\\n`` and ``\\t``)."""
    text = text.translate(_ZERO_WIDTH)
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or (unicodedata.category(ch)[0] != "C" and ch != " " and ch != " ")
    )


def terms(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lower-cased word pieces; ``CamelCase`` and ``snake_case`` identifiers are split."""
    out: list[str] = []
    for tok in _WORD.findall(text.replace("_", " ")):
        for piece in _CAMEL.split(tok):
            p = piece.lower()
            if len(p) < 2 and not p.isdigit():
                continue
            if not keep_stopwords and p in STOPWORDS:
                continue
            out.append(p)
    return out


def term_set(text: str) -> set[str]:
    return set(terms(text))


def truncate_at_boundary(text: str, limit: int) -> tuple[str, bool]:
    """Cut ``text`` to <= ``limit`` characters at a paragraph/line/sentence boundary when possible."""
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    for sep in ("\n\n", "\n", ". "):
        pos = cut.rfind(sep)
        if pos >= limit * 0.6:
            cut = cut[: pos + len(sep)].rstrip()
            break
    return cut.rstrip() + " [...]", True
