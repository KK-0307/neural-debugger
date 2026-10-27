"""Append-only storage for user feedback on answers (Milestone 6).

Deliberately small: one JSON object per line in ``<dir>/feedback.jsonl``. What is stored is decided by the API layer: the
rating, reason tags, a short redacted comment, and the question text **only if the user ticked the opt-in box**.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 20 * 1024 * 1024  # stop accepting feedback rather than grow without bound

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\"?\s*[:=]\s*\"?[A-Za-z0-9._\-]{8,}"),
]


class FeedbackStorageFullError(RuntimeError):
    pass


def redact_secrets(text: str) -> str:
    """Remove key-looking strings from user-supplied free text before it is written to disk."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class FeedbackStore:
    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / "feedback.jsonl"
        self._lock = threading.Lock()

    def add(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size + len(line) > MAX_FILE_BYTES:
                raise FeedbackStorageFullError(str(self.path))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text(encoding="utf-8").splitlines() if ln]
