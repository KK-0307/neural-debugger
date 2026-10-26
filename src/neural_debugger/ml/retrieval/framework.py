"""Framework awareness: a corpus-derived query classifier and a router that uses it.

Cross-framework confusion is one of BM25's documented failure modes (a FastAPI question retrieving
Python's exception docs). Users can pick a framework explicitly (``framework=`` filter); when they
do not, ``FrameworkDetector`` estimates which documentation set the query is about.

The detector is a multinomial naive Bayes model trained **only on the documentation corpus** (no
evaluation queries), with a uniform prior so Python's large share of the corpus does not dominate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from neural_debugger.data.schema import Chunk

from .tokenizer import TokenizerConfig, tokenize
from .types import Retriever, SearchHit

FRAMEWORKS = ("fastapi", "python", "pytorch")
_TOK = TokenizerConfig()


class FrameworkDetector:
    def __init__(self, chunks: Sequence[Chunk], smoothing: float = 0.1) -> None:
        self.smoothing = smoothing
        self.counts: dict[str, Counter[str]] = {f: Counter() for f in FRAMEWORKS}
        for c in chunks:
            self.counts[c.framework].update(tokenize(c.index_text(), _TOK))
        self.totals = {f: sum(cnt.values()) for f, cnt in self.counts.items()}
        self.vocab = set().union(*self.counts.values())

    def predict_proba(self, query: str) -> dict[str, float]:
        tokens = [t for t in tokenize(query, _TOK) if t in self.vocab]
        v = len(self.vocab)
        logp = {}
        for f in FRAMEWORKS:
            denom = self.totals[f] + self.smoothing * v
            logp[f] = sum(math.log((self.counts[f][t] + self.smoothing) / denom) for t in tokens)
        m = max(logp.values())
        exp = {f: math.exp(lp - m) for f, lp in logp.items()}
        z = sum(exp.values())
        return {f: exp[f] / z for f in FRAMEWORKS}

    def predict(self, query: str) -> tuple[str, float]:
        p = self.predict_proba(query)
        best = max(p, key=lambda f: p[f])
        return best, p[best]


class FrameworkRouter:
    """Wraps a retriever with automatic framework awareness.

    * ``hard``: if the detector's top probability >= ``threshold``, restrict search to that
      framework; otherwise search everything.
    * ``soft``: search everything, then re-rank with ``1/(60+rank) * (1 + beta * P(framework))``.
    An explicit ``framework`` argument always wins (a user-chosen filter is never overridden).
    """

    def __init__(
        self,
        base: Retriever,
        detector: FrameworkDetector,
        mode: str = "hard",
        threshold: float = 0.9,
        beta: float = 1.0,
        depth: int = 100,
        name: str = "router",
    ) -> None:
        if mode not in {"hard", "soft"}:
            raise ValueError("mode must be 'hard' or 'soft'")
        self.name, self.base, self.detector = name, base, detector
        self.mode, self.threshold, self.beta, self.depth = mode, threshold, beta, depth

    def search(self, query: str, k: int = 10, framework: str | None = None) -> list[SearchHit]:
        if framework is not None:
            return self.base.search(query, k, framework)
        proba = self.detector.predict_proba(query)
        if self.mode == "hard":
            best = max(proba, key=lambda f: proba[f])
            return self.base.search(query, k, best if proba[best] >= self.threshold else None)
        hits = self.base.search(query, max(self.depth, k), None)
        scored = [((1.0 / (60 + h.rank)) * (1.0 + self.beta * proba[h.framework]), h) for h in hits]
        scored.sort(key=lambda sh: (-sh[0], sh[1].rank))  # ties keep the base order
        return [replace(h, rank=i + 1, score=round(s, 8)) for i, (s, h) in enumerate(scored[:k])]
