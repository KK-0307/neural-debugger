"""Relevance judgments ("qrels") with explicit tracking of *which sections were ever judged*.

The labels were pooled from BM25's top-8 only. A system that retrieves a section nobody judged is
indistinguishable from one that retrieved an irrelevant section, which biases comparisons toward
BM25. So every judgment set records, per query, the sections that have been judged (relevant or
not), and results report:

* ``unjudged@10``: share of a system's top-10 sections that were never judged (coverage gap);
* condensed-list metrics (Sakai 2007): the ranking with unjudged sections removed, then scored;
* results under ``original`` qrels (frozen) and ``expanded`` qrels (original + neural pool review).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from neural_debugger.data.corpus import Corpus
from neural_debugger.data.queries import QueryRecord, expand_section_id
from neural_debugger.ml.retrieval.bm25 import BM25Config, BM25Retriever

NEURAL_POOL_FILES = ("pooled_judgments_neural.yaml", "pooled_judgments_neural_test.yaml")
# Milestone 4 adds its own file so the M3-frozen judgments (and the numbers derived from them) stay as they were.
M4_POOL_FILES = ("pooled_judgments_m4.yaml", "pooled_judgments_m4_test.yaml")
BM25_POOL_DEPTH = 8
Variant = Literal["original", "expanded", "m4"]


@dataclass
class Qrels:
    relevant: dict[str, set[str]]
    judged: dict[str, set[str]]


def bm25_pool_judged(corpus: Corpus, queries: Sequence[QueryRecord]) -> dict[str, set[str]]:
    """Sections reviewed during the Milestone 2 pooled review (BM25 top-8, framework-filtered).

    Recomputed with the frozen BM25 configuration, exactly as ``scripts/pool_candidates.py`` did;
    the official-FAQ slice was never pooled, so only its answer section counts as judged.
    """
    bm25 = BM25Retriever(corpus.chunks, BM25Config())
    judged: dict[str, set[str]] = {}
    for q in queries:
        s = set(q.relevant)
        if q.slice != "official_faq":
            s |= {
                h.section_id for h in bm25.search(q.query, k=BM25_POOL_DEPTH, framework=q.framework)
            }
        judged[q.id] = s
    return judged


def load_qrels(
    queries: Sequence[QueryRecord],
    corpus: Corpus,
    variant: Variant = "original",
    eval_dir: Path = Path("data/eval"),
) -> Qrels:
    relevant = {q.id: set(q.relevant) for q in queries}
    judged = bm25_pool_judged(corpus, queries)
    if variant in ("expanded", "m4"):
        files = NEURAL_POOL_FILES + (M4_POOL_FILES if variant == "m4" else ())
        for fname in files:
            path = eval_dir / fname
            if not path.exists():
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for qid, entry in data.items():
                if qid not in relevant:
                    continue
                added = {expand_section_id(s) for s in entry.get("added", [])}
                reviewed = {expand_section_id(s) for s in entry.get("reviewed", [])}
                relevant[qid] |= added
                judged[qid] |= reviewed | added
    return Qrels(relevant, judged)


def unjudged_at_k(ranking: Sequence[str], judged: set[str], k: int = 10) -> float:
    top = ranking[:k]
    return sum(s not in judged for s in top) / len(top) if top else 0.0


def condense(ranking: Sequence[str], judged: set[str]) -> list[str]:
    """Remove never-judged sections from a ranking (condensed-list evaluation)."""
    return [s for s in ranking if s in judged]


def coverage_summary(per_query: Mapping[str, float]) -> float:
    return sum(per_query.values()) / len(per_query) if per_query else 0.0
