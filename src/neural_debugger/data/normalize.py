"""Text normalisation.

Prose is normalised aggressively enough to make matching and de-duplication reliable;
code is touched as little as possible so identifiers, indentation and literals survive.
"""

from __future__ import annotations

import re
import unicodedata

_PROSE_TRANSLATE = str.maketrans(
    {
        " ": " ",  # no-break space
        "​": "",  # zero-width space
        "‌": "",
        "‍": "",
        "﻿": "",
        "­": "",  # soft hyphen
        "¶": "",  # pilcrow used for Sphinx permalinks
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "…": "...",
    }
)
_CODE_TRANSLATE = str.maketrans({" ": " ", "​": "", "﻿": "", "­": ""})
_SPACES = re.compile(r"[ \t]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_prose(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_PROSE_TRANSLATE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", ln).strip() for ln in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def normalize_code(text: str) -> str:
    """Keep indentation and content; only fix line endings, trailing space and blank edges."""
    text = unicodedata.normalize("NFC", text).translate(_CODE_TRANSLATE)
    text = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def dedup_key(text: str) -> str:
    """Case/whitespace-insensitive key for exact-duplicate detection."""
    return _SPACES.sub(" ", re.sub(r"\s+", " ", text)).strip().lower()


def slugify(heading: str) -> str:
    """mkdocs/python-markdown ``toc`` style slug, used only when a heading has no explicit id."""
    value = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)
