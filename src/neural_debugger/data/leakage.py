"""Dataset integrity and leakage checks (run by ``dataset check`` and by the test-suite)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .corpus import Corpus
from .queries import QueryRecord, validate_queries
from .splits import (
    NEAR_DUPLICATE_JACCARD,
    assign_splits,
    near_duplicate_pairs,
    normalize_query,
)


@dataclass
class LeakageReport:
    errors: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_dataset(queries: list[QueryRecord], corpus: Corpus) -> LeakageReport:
    report = LeakageReport()
    report.errors.extend(validate_queries(queries, corpus))
    by_id = {q.id: q for q in queries}

    missing = [q.id for q in queries if q.split is None]
    if missing:
        report.errors.append(f"queries without a split: {missing[:5]}")
        return report

    # 1. exact + near duplicates may never straddle partitions
    for a, b, sim in near_duplicate_pairs(queries, NEAR_DUPLICATE_JACCARD):
        if by_id[a].split != by_id[b].split:
            report.errors.append(
                f"near-duplicate queries in different splits: {a} ({by_id[a].split}) / "
                f"{b} ({by_id[b].split}) jaccard={sim}"
            )
    texts = Counter(normalize_query(q.query) for q in queries)
    for text, n in texts.items():
        if n > 1:
            report.errors.append(f"identical query text appears {n} times: {text[:60]!r}")

    # 2. stored splits must equal the deterministic assignment
    expected = assign_splits(queries)
    drift = [qid for qid, s in expected.items() if by_id[qid].split != s]
    if drift:
        report.errors.append(f"stored split differs from deterministic assignment for {drift[:5]}")

    # 3. every partition is usable
    counts = Counter(q.split for q in queries)
    for split in ("train", "val", "test"):
        if counts[split] == 0:
            report.errors.append(f"empty split: {split}")

    # 4. statistics (informational): how much do test targets overlap training targets?
    train_sections = {s for q in queries if q.split == "train" for s in q.relevant}
    test_queries = [q for q in queries if q.split == "test"]
    test_sections = {s for q in test_queries for s in q.relevant}
    index_text = " ".join(" ".join(c.index_text().lower().split()) for c in corpus.chunks)
    verbatim: dict[str, list[int]] = {}
    for q in queries:
        hit = normalize_query(q.query) in index_text
        verbatim.setdefault(q.slice, []).append(int(hit))
    report.stats = {
        "n_queries": len(queries),
        "by_split": dict(counts),
        "by_slice": dict(Counter(q.slice for q in queries)),
        "by_framework": dict(Counter(q.framework for q in queries)),
        "test_relevant_sections_also_relevant_in_train": (
            round(len(test_sections & train_sections) / len(test_sections), 3)
            if test_sections
            else None
        ),
        "queries_appearing_verbatim_in_corpus_text": {
            k: f"{sum(v)}/{len(v)}" for k, v in sorted(verbatim.items())
        },
    }
    return report
