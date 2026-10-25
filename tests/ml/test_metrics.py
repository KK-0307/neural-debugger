import math
import random

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from neural_debugger.ml.evaluation.metrics import (
    bootstrap_ci,
    hit_at_k,
    ndcg_at_k,
    query_metrics,
    recall_at_k,
    reciprocal_rank,
    summarize,
)

RANKED = ["a", "b", "c", "d"]
REL = {"c", "e"}  # "e" is never retrieved


def test_hand_computed_values() -> None:
    assert recall_at_k(RANKED, REL, 1) == 0.0
    assert recall_at_k(RANKED, REL, 3) == 0.5
    assert recall_at_k(RANKED, REL, 10) == 0.5  # e was not retrieved
    assert hit_at_k(RANKED, REL, 2) == 0.0 and hit_at_k(RANKED, REL, 3) == 1.0
    assert reciprocal_rank(RANKED, REL) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, REL, k=2) == 0.0
    dcg = 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, REL, 10) == pytest.approx(dcg / idcg)


def test_perfect_and_empty_rankings() -> None:
    assert ndcg_at_k(["x", "y", "z"], {"x", "y"}, 10) == pytest.approx(1.0)
    assert query_metrics(["x"], {"x"}) == {
        "recall@1": 1.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "hit@1": 1.0,
        "hit@5": 1.0,
        "hit@10": 1.0,
        "mrr": 1.0,
        "ndcg@10": 1.0,
    }
    assert all(v == 0.0 for v in query_metrics([], {"x"}).values())


def test_multiple_relevant_recall_at_1_is_a_fraction() -> None:
    assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5
    assert hit_at_k(["a", "b"], {"a", "b"}, 1) == 1.0


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="no relevant"):
        recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError, match="duplicate"):
        ndcg_at_k(["a", "a"], {"a"}, 10)


def test_ndcg_matches_scikit_learn_on_random_cases() -> None:
    rng = random.Random(7)
    for _ in range(60):
        universe = [f"s{i}" for i in range(40)]
        ranked = rng.sample(universe, 25)  # retrieved list (>= k so no tie ambiguity at k=10)
        relevant = set(rng.sample(universe, rng.randint(1, 6)))
        # scores: retrieved items get descending scores; unretrieved items score lowest
        score = {s: float(len(ranked) - i + 100) for i, s in enumerate(ranked)}
        y_score = np.array([[score.get(s, 0.0) for s in universe]])
        y_true = np.array([[1.0 if s in relevant else 0.0 for s in universe]])
        assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(
            ndcg_score(y_true, y_score, k=10), abs=1e-9
        )


def test_metric_bounds_and_monotonicity_in_k() -> None:
    rng = random.Random(1)
    for _ in range(100):
        universe = [f"s{i}" for i in range(30)]
        ranked = rng.sample(universe, 20)
        relevant = set(rng.sample(universe, rng.randint(1, 5)))
        m = query_metrics(ranked, relevant)
        assert all(0.0 <= v <= 1.0 for v in m.values())
        assert m["recall@1"] <= m["recall@5"] <= m["recall@10"]
        assert m["hit@1"] <= m["hit@5"] <= m["hit@10"]
        assert m["mrr"] >= m["hit@1"] * 1.0 / 1


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    vals = [0.0, 1.0, 1.0, 0.5, 0.25, 0.75, 1.0, 0.0]
    lo, hi = bootstrap_ci(vals, seed=3)
    assert (lo, hi) == bootstrap_ci(vals, seed=3)
    assert lo <= np.mean(vals) <= hi
    assert bootstrap_ci([0.7] * 10) == (0.7, 0.7)


def test_summarize_means_equal_manual_average() -> None:
    per_query = {"q1": query_metrics(["a"], {"a"}), "q2": query_metrics(["b", "a"], {"a"})}
    s = summarize(per_query)
    assert s.n_queries == 2 and s.mean["mrr"] == pytest.approx(0.75)
    assert s.mean["recall@1"] == pytest.approx(0.5)
