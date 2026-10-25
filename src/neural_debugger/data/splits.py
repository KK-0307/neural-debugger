"""Deterministic, leakage-aware train/val/test assignment.

* Queries are first grouped: identical or near-identical queries (token-set Jaccard >= 0.6) are
  placed in one group, and a group is never split across partitions.
* A group's partition depends only on a hash of its smallest member's normalised text and a fixed
  salt, so adding new queries never moves existing ones (unless it merges two groups).
* 60 / 20 / 20 by hash bucket: sizes are approximate, not forced.
"""

from __future__ import annotations

import hashlib
import re

from neural_debugger.ml.retrieval.tokenizer import STOPWORDS

from .queries import QueryRecord, Split

SPLIT_SALT = "neural-debugger-split-v1"
NEAR_DUPLICATE_JACCARD = 0.6
TRAIN_PCT, VAL_PCT = 60, 20  # remainder (20) is test
_WORD = re.compile(r"[a-z0-9]+")


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


def token_set(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicate_pairs(
    queries: list[QueryRecord], threshold: float = NEAR_DUPLICATE_JACCARD
) -> list[tuple[str, str, float]]:
    sets = [token_set(q.query) for q in queries]
    out = []
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            sim = jaccard(sets[i], sets[j])
            if sim >= threshold or normalize_query(queries[i].query) == normalize_query(
                queries[j].query
            ):
                out.append((queries[i].id, queries[j].id, round(sim, 3)))
    return out


def group_queries(
    queries: list[QueryRecord], threshold: float = NEAR_DUPLICATE_JACCARD
) -> list[list[str]]:
    parent = {q.id: q.id for q in queries}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in near_duplicate_pairs(queries, threshold):
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for q in queries:
        groups.setdefault(find(q.id), []).append(q.id)
    return sorted(groups.values())


def split_for_group_key(key: str) -> Split:
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).hexdigest()[:8], 16) % 100
    if bucket < TRAIN_PCT:
        return "train"
    if bucket < TRAIN_PCT + VAL_PCT:
        return "val"
    return "test"


def assign_splits(queries: list[QueryRecord]) -> dict[str, Split]:
    by_id = {q.id: q for q in queries}
    result: dict[str, Split] = {}
    for group in group_queries(queries):
        key = min(
            hashlib.sha256(normalize_query(by_id[i].query).encode()).hexdigest() for i in group
        )
        split = split_for_group_key(key)
        for qid in group:
            result[qid] = split
    return result
