"""Select, de-duplicate, sanitise and budget retrieved sections into ``Evidence``."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .injection import REMOVED_LINE, neutralize_document
from .text import term_set, truncate_at_boundary
from .types import Evidence, RetrievedSection

_VERSION_NOTE = re.compile(
    r"((?:Added|Changed|Deprecated|Removed|New)\s+(?:in|since)\s+version\s+[\w.]+[^\n]{0,140})",
    re.I,
)


@dataclass(frozen=True)
class EvidenceConfig:
    max_items: int = 5
    max_chars_each: int = 1500
    max_chars_total: int = 6000
    max_per_doc: int = 2
    near_duplicate_jaccard: float = 0.8
    min_text_chars: int = 40
    drop_if_removed_fraction: float = (
        0.5  # of lines removed as instruction-like -> drop the passage
    )


@dataclass
class SelectionLog:
    dropped: list[dict[str, str]] = field(default_factory=list)

    def drop(self, section_id: str, reason: str) -> None:
        self.dropped.append({"section_id": section_id, "reason": reason})


def _fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def version_notes(text: str, limit: int = 3) -> tuple[str, ...]:
    notes: list[str] = []
    for m in _VERSION_NOTE.finditer(text):
        note = " ".join(m.group(1).split())
        if note not in notes:
            notes.append(note)
        if len(notes) >= limit:
            break
    return tuple(notes)


def _doc_of(section_id: str) -> str:
    return section_id.split("#", 1)[0]


def select_evidence(
    sections: list[RetrievedSection], cfg: EvidenceConfig | None = None
) -> tuple[list[Evidence], SelectionLog]:
    cfg = cfg or EvidenceConfig()
    log = SelectionLog()
    kept: list[Evidence] = []
    kept_terms: list[set[str]] = []
    seen_fp: set[str] = set()
    per_doc: dict[str, int] = {}
    total = 0
    for sec in sorted(sections, key=lambda s: s.rank):
        if len(kept) >= cfg.max_items:
            break
        if not sec.text.strip():
            log.drop(sec.section_id, "missing_text")
            continue
        text, findings = neutralize_document(sec.text)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        removed = sum(ln == REMOVED_LINE for ln in lines)
        if lines and removed / len(lines) >= cfg.drop_if_removed_fraction:
            log.drop(sec.section_id, "instruction_like_content")
            continue
        text = text.strip()
        if len(text) < cfg.min_text_chars:
            log.drop(sec.section_id, "too_short")
            continue
        fp = _fingerprint(text)
        ts = term_set(text)
        if fp in seen_fp or any(_jaccard(ts, o) >= cfg.near_duplicate_jaccard for o in kept_terms):
            log.drop(sec.section_id, "duplicate")
            continue
        doc = _doc_of(sec.section_id)
        if per_doc.get(doc, 0) >= cfg.max_per_doc:
            log.drop(sec.section_id, "doc_cap")
            continue
        notes = version_notes(text)
        text, _ = truncate_at_boundary(text, cfg.max_chars_each)
        if total + len(text) > cfg.max_chars_total:
            room = cfg.max_chars_total - total
            if room < 300:
                log.drop(sec.section_id, "budget")
                break
            text, _ = truncate_at_boundary(text, room)
        total += len(text)
        seen_fp.add(fp)
        kept_terms.append(ts)
        per_doc[doc] = per_doc.get(doc, 0) + 1
        flags = tuple(sorted({f"injection:{f.rule}" for f in findings}))
        kept.append(
            Evidence(
                id=f"E{len(kept) + 1}",
                section_id=sec.section_id,
                framework=sec.framework,
                version=sec.version,
                title=sec.title,
                heading=sec.heading,
                url=sec.url,
                text=text,
                rank=sec.rank,
                score=sec.score,
                flags=flags,
                version_notes=notes,
            )
        )
    return kept, log
