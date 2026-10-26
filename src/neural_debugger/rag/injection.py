"""Defences against prompt injection in retrieved documentation and in user-submitted text.

Threat model: text that reaches the prompt may contain instructions aimed at the model (a poisoned page,
a pasted log, a forum answer copied into a question). The defence is layered; no single layer is claimed
to be sufficient:

1. **Structure** (``prompt.py``): untrusted text only ever appears inside delimited data blocks, the
   system prompt states that block contents are data and never instructions, and delimiter look-alikes are
   escaped here so a document cannot close its own block.
2. **Scanning** (this module): instruction-like lines in *documentation* are removed and the passage is
   flagged; in the *user's* text they are only flagged (the user may legitimately paste anything, and the
   model is told to treat it as data).
3. **Output validation** (``validate.py``): whatever the model says, citations must name evidence it was
   shown, URLs must be among the source URLs, images/HTML are stripped, and code is never executed. A
   fully hijacked model therefore cannot invent a source or smuggle an exfiltration link into the answer.

The patterns are heuristics tuned for precision on the indexed documentation (see
``docs/rag.md`` for the measured false-positive rate on the corpus) and will miss paraphrased attacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .text import strip_invisible

Severity = Literal["high", "medium"]
REMOVED_LINE = "[line removed: instruction-like text]"


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    excerpt: str


_I = re.IGNORECASE
_RULES: tuple[tuple[str, Severity, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        "high",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass|drop)\b[^.\n]{0,40}\b(previous|prior|above|earlier|"
            r"preceding|all|any|your|the|these|those|system)\b[^.\n]{0,30}\b(instructions?|rules?|prompts?|"
            r"guidelines?|directions?|constraints?|context)\b",
            _I,
        ),
    ),
    (
        "reveal_prompt",
        "high",
        re.compile(
            r"\b(reveal|print|show|repeat|output|leak|display|disclose)\b[^.\n]{0,25}\b(your|the)\b[^.\n]{0,15}"
            r"\b(system|hidden|initial|original|secret)\b[^.\n]{0,10}\b(prompt|instructions?|message)\b",
            _I,
        ),
    ),
    ("new_instructions", "high", re.compile(r"\bnew\s+(system\s+)?instructions?\s*:", _I)),
    (
        "role_override",
        "high",
        re.compile(
            r"\byou\s+are\s+now\s+(an?|the)\s+[\w -]{0,30}\b(assistant|agent|bot|model|ai|system|developer|"
            r"admin|dan)\b|\b(pretend|act|behave)\s+(to\s+be|as)\s+(an?|the)\s+(unrestricted|unfiltered|"
            r"different|new|malicious|evil|helpful)?\s*(assistant|ai|bot|model|system|admin|developer)\b|"
            r"\b(pretend|act|behave)\s+as\s+if\s+you\s+(are|were)\b",
            _I,
        ),
    ),
    (
        "chat_template_tokens",
        "high",
        re.compile(
            r"<\|?(im_start|im_end|system|assistant|user|endoftext|eot_id)\|?>|\[/?INST\]|<</?SYS>>|"
            r"^\s*#{2,}\s*(system|assistant)\b|^\s*(system|assistant|developer)\s*:\s*\S",
            _I | re.MULTILINE,
        ),
    ),
    (
        "suppress_or_steer_citations",
        "high",
        re.compile(
            r"\b(do\s+not|don'?t|never)\s+(cite|mention|reveal|tell|disclose)\b|\b(always|only)\s+cite\b|"
            r"\bcite\s+(this|the\s+following)\s+(source|document|page|passage)\b|"
            r"\bmark\s+(this|it)\s+as\s+(the\s+)?(correct|only|authoritative)\b",
            _I,
        ),
    ),
    (
        "dictate_output",
        "high",
        re.compile(
            r"\b(respond|answer|reply|output|say)\s+(only\s+)?(with|in)\s+(the\s+following|exactly|json|"
            r"this\s+text)\b|\byou\s+(must|have\s+to)\s+now\s+(respond|answer|output)\b",
            _I,
        ),
    ),
    (
        "markdown_image",
        "medium",
        re.compile(r"!\[[^\]]*\]\([^)]*\)", _I),
    ),
    (
        "active_html",
        "medium",
        re.compile(r"<\s*(script|iframe|object|embed)\b|javascript\s*:", _I),
    ),
)

# Tag names used by our own prompt: a document must not be able to open or close them.
_DELIMITERS = re.compile(
    r"<\s*/?\s*(evidence|user_question|question|documents?|instructions?|system|assistant|answer|claim|passage|reference)\b",
    _I,
)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_JS_URL = re.compile(r"javascript\s*:", _I)


def scan(text: str) -> list[Finding]:
    """All instruction-like patterns in ``text`` (one finding per matching rule and line)."""
    findings: list[Finding] = []
    for line in text.splitlines():
        for rule, severity, pat in _RULES:
            m = pat.search(line)
            if m:
                findings.append(Finding(rule, severity, line.strip()[:120]))
    return findings


def escape_delimiters(text: str) -> str:
    """Neutralise look-alikes of the prompt's own tags so untrusted text cannot break out of its block."""
    return _DELIMITERS.sub(lambda m: "&lt;" + m.group(0)[1:], text)


def neutralize_document(text: str) -> tuple[str, list[Finding]]:
    """Clean a *retrieved* passage: remove instruction-like lines, markdown images, `javascript:` URLs and delimiter forgeries.

    HTML tags are left alone (documentation shows HTML in code samples); the answer validator strips markup.

    Returns the cleaned text and the findings (so the caller can flag or drop the passage).
    """
    text = strip_invisible(text)
    findings: list[Finding] = []
    kept: list[str] = []
    for line in text.splitlines():
        hit = [
            Finding(rule, sev, line.strip()[:120])
            for rule, sev, pat in _RULES
            if sev == "high" and pat.search(line)
        ]
        if hit:
            findings.extend(hit)
            kept.append(REMOVED_LINE)
        else:
            kept.append(line)
    cleaned = "\n".join(kept)
    for rule, sev, pat in _RULES:
        if sev == "medium" and pat.search(cleaned):
            findings.append(Finding(rule, sev, pat.search(cleaned).group(0)[:120]))  # type: ignore[union-attr]
    cleaned = _MD_IMAGE.sub("[image removed]", cleaned)
    cleaned = _JS_URL.sub("[removed]:", cleaned)
    return escape_delimiters(cleaned), findings


def sanitize_user_text(text: str) -> tuple[str, list[Finding]]:
    """Normalise user-submitted text. Only flags instruction-like content; it is still data, not removed."""
    text = strip_invisible(text)
    return escape_delimiters(text), scan(text)
