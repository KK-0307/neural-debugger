"""Ranking metrics with binary relevance, computed over *sections*.

Conventions (all documented in docs/evaluation.md):

* ``ranked`` is a best-first list of unique section ids; ``relevant`` the non-empty set of
  relevant section ids for the query.
* ``recall@k``  = |relevant ∩ top-k| / |relevant|
* ``hit@k``     = 1 if any relevant section is in the top-k else 0
* ``MRR``       = 1 / rank of the first relevant section within the ranking (0 if absent)
* ``ndcg@k``    = DCG@k / IDCG@k, gain 1 for a relevant section, discount 1/log2(rank+1)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def _check(ranked: Sequence[str], relevant: set[str]) -> None:
    if not relevant:
        raise ValueError("query has no relevant sections")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranking contains duplicate sections; collapse to sections first")


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranked, relevant)
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranked, relevant)
    return 1.0 if relevant.intersection(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    _check(ranked, relevant)
    for i, section in enumerate(ranked[:k], start=1):
        if section in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranked, relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, s in enumerate(ranked[:k], start=1) if s in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


METRIC_NAMES = ("recall@1", "recall@5", "recall@10", "hit@1", "hit@5", "hit@10", "mrr", "ndcg@10")


def query_metrics(ranked: Sequence[str], relevant: set[str]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked, relevant, 1),
        "recall@5": recall_at_k(ranked, relevant, 5),
        "recall@10": recall_at_k(ranked, relevant, 10),
        "hit@1": hit_at_k(ranked, relevant, 1),
        "hit@5": hit_at_k(ranked, relevant, 5),
        "hit@10": hit_at_k(ranked, relevant, 10),
        "mrr": reciprocal_rank(ranked, relevant),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
    }


@dataclass(frozen=True)
class MetricSummary:
    n_queries: int
    mean: dict[str, float]
    ci95: dict[str, tuple[float, float]]  # percentile bootstrap over queries

    def as_dict(self) -> dict[str, object]:
        return {
            "n_queries": self.n_queries,
            "mean": {k: round(v, 4) for k, v in self.mean.items()},
            "ci95": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in self.ci95.items()},
        }


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = arr[rng.randint(0, arr.size, size=(n_resamples, arr.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize(per_query: Mapping[str, Mapping[str, float]], seed: int = 0) -> MetricSummary:
    names = list(next(iter(per_query.values())).keys()) if per_query else list(METRIC_NAMES)
    mean: dict[str, float] = {}
    ci: dict[str, tuple[float, float]] = {}
    for name in names:
        vals = [m[name] for m in per_query.values()]
        mean[name] = float(np.mean(vals)) if vals else float("nan")
        ci[name] = bootstrap_ci(vals, seed=seed)
    return MetricSummary(len(per_query), mean, ci)
