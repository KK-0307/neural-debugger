"""Decide, *before* calling the model, whether the retrieved evidence can plausibly answer the question.

The signal is IDF-weighted coverage of the question's salient terms by the evidence (titles, headings and
text). A term that is not in the indexed documentation at all (``tensorflow``, ``django``) gets the maximum
weight, so a question about an unindexed library has low coverage even when some generic words match.
Thresholds are calibrated on development queries only (``docs/rag.md``); they are heuristics, not
probabilities.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from .text import term_set, terms
from .types import Evidence, QuestionInfo

Level = Literal["none", "weak", "ok"]


class TermStats:
    """Document frequencies over the corpus' chunks (section-level units are fine for IDF)."""

    def __init__(self, df: dict[str, int], n_docs: int) -> None:
        self.df, self.n = df, max(n_docs, 1)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> TermStats:
        df: Counter[str] = Counter()
        n = 0
        for t in texts:
            n += 1
            df.update(term_set(t))
        return cls(dict(df), n)

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        return math.log((self.n + 1) / (d + 0.5)) + 1.0  # unknown term -> largest weight


class UniformStats(TermStats):
    def __init__(self) -> None:
        super().__init__({}, 1)

    def idf(self, term: str) -> float:  # noqa: ARG002
        return 1.0


@dataclass(frozen=True)
class SufficiencyConfig:
    abstain_below: float = 0.375  # calibrated on dev, see docs/rag.md
    confident_above: float = 0.65
    max_terms: int = 40


@dataclass
class Sufficiency:
    level: Level
    coverage: float
    missing_terms: list[str] = field(default_factory=list)
    reason: str = ""


def salient_terms(q: QuestionInfo, limit: int) -> list[str]:
    seen: list[str] = []
    for t in terms(" ".join([*q.error_lines[-1:], q.gate_text or q.retrieval_query])):
        if t not in seen and not t.isdigit():
            seen.append(t)
    return seen[:limit]


def assess(
    q: QuestionInfo,
    evidence: list[Evidence],
    stats: TermStats | None = None,
    cfg: SufficiencyConfig | None = None,
) -> Sufficiency:
    cfg = cfg or SufficiencyConfig()
    stats = stats or UniformStats()
    if not evidence:
        return Sufficiency("none", 0.0, [], "no usable documentation was retrieved")
    wanted = salient_terms(q, cfg.max_terms)
    if not wanted:
        return Sufficiency("weak", 0.0, [], "the question has no searchable terms")
    have: set[str] = set()
    for e in evidence:
        have |= term_set(" ".join([e.title, e.heading, e.text]))
    total = sum(stats.idf(t) for t in wanted)
    covered = sum(stats.idf(t) for t in wanted if t in have)
    cov = covered / total if total else 0.0
    missing = sorted((t for t in wanted if t not in have), key=lambda t: -stats.idf(t))[:8]
    if cov < cfg.abstain_below:
        return Sufficiency(
            "none",
            cov,
            missing,
            f"retrieved documentation covers only {cov:.0%} of the question's key terms",
        )
    if cov < cfg.confident_above:
        return Sufficiency("weak", cov, missing, f"partial coverage ({cov:.0%}) of the key terms")
    return Sufficiency("ok", cov, missing, "")
