"""Chunk de-duplication: exact (normalised text) and near-duplicate (MinHash-LSH + exact Jaccard).

The first chunk in corpus order is kept; later copies are dropped and recorded in the
report, so the decision is auditable and deterministic.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .normalize import dedup_key
from .schema import Chunk

_WORD = re.compile(r"\w+")
_PRIME = (1 << 31) - 1


def shingles(text: str, n: int = 5) -> set[int]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return {
        int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big") % _PRIME
        for g in grams
    }


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, _PRIME, size=num_perm).astype(np.int64)
        self.b = rng.randint(0, _PRIME, size=num_perm).astype(np.int64)

    def signature(self, sh: set[int]) -> np.ndarray:
        if not sh:
            return np.full(len(self.a), _PRIME, dtype=np.int64)
        h = np.fromiter(sh, dtype=np.int64)
        return ((self.a[:, None] * h[None, :] + self.b[:, None]) % _PRIME).min(axis=1)


def near_duplicate_pairs(
    sets: list[set[int]], threshold: float, bands: int = 32, num_perm: int = 128
) -> list[tuple[int, int, float]]:
    """All index pairs (i < j) whose exact Jaccard similarity is >= threshold (LSH candidates)."""
    hasher = MinHasher(num_perm)
    sigs = [hasher.signature(s) for s in sets]
    rows = num_perm // bands
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for band in range(bands):
            buckets[(band, sig[band * rows : (band + 1) * rows].tobytes())].append(idx)
    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                candidates.add((members[x], members[y]))
    out = []
    for i, j in sorted(candidates):
        sim = jaccard(sets[i], sets[j])
        if sim >= threshold:
            out.append((i, j, round(sim, 4)))
    return out


@dataclass
class DedupReport:
    n_input: int = 0
    n_exact_duplicates: int = 0
    n_near_duplicates: int = 0
    near_threshold: float = 0.85
    dropped: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "n_input": self.n_input,
            "n_exact_duplicates_dropped": self.n_exact_duplicates,
            "n_near_duplicates_dropped": self.n_near_duplicates,
            "near_duplicate_jaccard_threshold": self.near_threshold,
            "n_output": self.n_input - self.n_exact_duplicates - self.n_near_duplicates,
        }


def dedupe_chunks(
    chunks: list[Chunk], near_threshold: float = 0.85, min_words_for_near: int = 25
) -> tuple[list[Chunk], DedupReport]:
    report = DedupReport(n_input=len(chunks), near_threshold=near_threshold)
    kept: list[Chunk] = []
    seen_exact: dict[str, str] = {}
    for c in chunks:
        key = dedup_key(c.content)
        if key in seen_exact:
            report.n_exact_duplicates += 1
            report.dropped.append(
                {
                    "dropped": c.chunk_id,
                    "duplicate_of": seen_exact[key],
                    "kind": "exact",
                    "jaccard": 1.0,
                }
            )
            continue
        seen_exact[key] = c.chunk_id
        kept.append(c)

    eligible = [i for i, c in enumerate(kept) if c.n_words >= min_words_for_near]
    sets = [shingles(kept[i].content) for i in eligible]
    drop: set[int] = set()
    for a, b, sim in near_duplicate_pairs(sets, near_threshold):
        ia, ib = eligible[a], eligible[b]  # ia < ib: keep the earlier chunk
        if ia in drop or ib in drop:
            continue
        drop.add(ib)
        report.dropped.append(
            {
                "dropped": kept[ib].chunk_id,
                "duplicate_of": kept[ia].chunk_id,
                "kind": "near",
                "jaccard": sim,
            }
        )
    report.n_near_duplicates = len(drop)
    return [c for i, c in enumerate(kept) if i not in drop], report
