"""Turn a raw user question (often a pasted traceback) into a ``QuestionInfo``.

Nothing here executes or evaluates user content: it is only normalised, length-limited, scanned and
searched with regular expressions.
"""

from __future__ import annotations

import re

from .injection import sanitize_user_text, scan
from .text import strip_invisible
from .types import QuestionInfo

MAX_CHARS = 6000
MAX_RETRIEVAL_CHARS = 400

_ERROR_LINE = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*\.)?[A-Za-z_]\w*(?:Error|Exception|Warning|Exit|Interrupt|Disconnect)\b.*$"
)
_TRACE_FRAME = re.compile(r'^\s*File ".*", line \d+')
_VERSION_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"\b(?:python|py)\s*[=v]*\s*(\d+\.\d+(?:\.\d+)?)", re.I),
    "pytorch": re.compile(
        r"\b(?:pytorch|torch)\s*(?:==|=|v|version\s+)?\s*(\d+\.\d+(?:\.\d+)?)", re.I
    ),
    "fastapi": re.compile(r"\bfastapi\s*(?:==|=|v|version\s+)?\s*(\d+\.\d+(?:\.\d+)?)", re.I),
}
_FRAMEWORK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pytorch": ("torch", "pytorch", "cuda", "tensor", "nn.module", "autograd", "dataloader", "mps"),
    "fastapi": (
        "fastapi",
        "pydantic",
        "starlette",
        "uvicorn",
        "apirouter",
        "depends(",
        "@app.get",
        "@app.post",
    ),
}


def _normalise(text: str) -> str:
    text = strip_invisible(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()


def _limit(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep the head and (more of) the tail: the exception line is at the end of a traceback."""
    if len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.35)
    tail = max_chars - head
    return text[:head].rstrip() + "\n[... truncated ...]\n" + text[-tail:].lstrip(), True


def extract_error_lines(text: str, limit: int = 3) -> tuple[str, ...]:
    seen: list[str] = []
    for line in text.splitlines():
        if _ERROR_LINE.match(line) and not _TRACE_FRAME.match(line):
            line = line.strip()[:300]
            if line not in seen:
                seen.append(line)
    return tuple(seen[-limit:])


def detect_framework_hint(text: str) -> str | None:
    low = text.lower()
    scores = {fw: sum(kw in low for kw in kws) for fw, kws in _FRAMEWORK_KEYWORDS.items()}
    best = max(scores.values())
    winners = [fw for fw, s in scores.items() if s == best]
    return winners[0] if best >= 1 and len(winners) == 1 else None


def extract_versions(text: str) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for fw, pat in _VERSION_PATTERNS.items():
        for m in pat.finditer(text):
            item = (fw, m.group(1))
            if item not in found:
                found.append(item)
    return tuple(found)


_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_FENCED = re.compile(r"```.*?(```|\Z)", re.S)


def _prose(text: str) -> str:
    """The user's own words about the problem.

    Dropped: fenced code (variable names and paths are not documentation terms), traceback frames and the
    source lines that follow them, and instruction-like sentences (they say nothing about the bug and must
    not steer retrieval).
    """
    keep: list[str] = []
    skip_next = False
    for line in _FENCED.sub("\n", text).splitlines():
        if any(f.severity == "high" for f in scan(line)):
            # drop only the offending sentences: the same line may also carry the user's real question
            line = " ".join(
                part
                for part in _SENTENCES.split(line)
                if not any(f.severity == "high" for f in scan(part))
            )
        if _TRACE_FRAME.match(line):
            skip_next = True
            continue
        if skip_next and line.startswith(("    ", "\t")):
            skip_next = False
            continue
        skip_next = False
        if line.strip() and not line.startswith("Traceback (most recent call last)"):
            keep.append(line.strip())
    return " ".join(keep)


def user_text_without_injection(text: str) -> str:
    """The question with instruction-like sentences removed; everything else (including code) is kept."""
    out: list[str] = []
    for line in text.splitlines():
        if any(f.severity == "high" for f in scan(line)):
            line = " ".join(
                part
                for part in _SENTENCES.split(line)
                if not any(f.severity == "high" for f in scan(part))
            )
        out.append(line)
    return "\n".join(out)


def prepare_question(raw: str, max_chars: int = MAX_CHARS) -> QuestionInfo:
    text = _normalise(raw)
    text, truncated = _limit(text, max_chars)
    clean, findings = sanitize_user_text(text)
    errors = extract_error_lines(clean)
    prose = _prose(clean)
    has_trace = any(_TRACE_FRAME.match(ln) for ln in clean.splitlines())
    if len(clean) <= MAX_RETRIEVAL_CHARS and not has_trace and "```" not in clean and not findings:
        query = clean.replace("\n", " ")
    else:
        head = errors[-1] if errors else ""
        rest = prose.replace(head, " ") if head else prose
        query = f"{head} {rest or clean}"
    return QuestionInfo(
        original=raw,
        text=clean,
        retrieval_query=query.strip()[:MAX_RETRIEVAL_CHARS],
        error_lines=errors,
        framework_hint=detect_framework_hint(clean),
        versions=extract_versions(clean),
        code_blocks=clean.count("```") // 2,
        truncated=truncated,
        injection_flags=tuple(sorted({f.rule for f in findings})),
        trusted_text=user_text_without_injection(clean),
    )
