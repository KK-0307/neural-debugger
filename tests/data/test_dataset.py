"""Unit tests for query loading, deterministic splits and leakage checks (small synthetic data)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml

from neural_debugger.data.corpus import Corpus
from neural_debugger.data.leakage import check_dataset
from neural_debugger.data.queries import QueryRecord, load_authored, validate_queries
from neural_debugger.data.splits import (
    assign_splits,
    group_queries,
    near_duplicate_pairs,
    split_for_group_key,
)
from tests.helpers import make_chunk

SECTION = "python/1.0/page#a"


def q(qid: str, text: str, split: str | None = None, rel: list[str] | None = None) -> QueryRecord:
    return QueryRecord(
        id=qid,
        query=text,
        slice="question",
        framework="python",
        authored_relevant=rel or [SECTION],
        split=split,
    )


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        chunks=[make_chunk("a", "text about a"), make_chunk("b", "text about b")],
        documents=[],
        sha256="x",
    )


def test_near_duplicates_are_grouped_and_distinct_queries_are_not() -> None:
    qs = [
        q("q1", "how do I read a file line by line in python"),
        q("q2", "How do I read a file line by line in Python?"),  # same words
        q("q3", "how to read a file line by line with python"),  # near duplicate
        q("q4", "create a virtual environment for my project"),
    ]
    groups = group_queries(qs)
    assert sorted(map(sorted, groups)) == [["q1", "q2", "q3"], ["q4"]]
    pairs = {(a, b) for a, b, _ in near_duplicate_pairs(qs)}
    assert ("q1", "q2") in pairs and not any("q4" in p for p in pairs)


def test_group_members_always_share_a_split() -> None:
    qs = [
        q(f"a{i}", f"how do I read file number {i} line by line in python today") for i in range(6)
    ]
    qs += [
        q(f"b{i}", f"completely different topic about cuda memory {i} and gpus on linux")
        for i in range(6)
    ]
    splits = assign_splits(qs)
    assert len({splits[f"a{i}"] for i in range(6)}) == 1
    assert len({splits[f"b{i}"] for i in range(6)}) == 1


def test_assignment_is_deterministic_and_order_independent() -> None:
    rng = random.Random(0)
    qs = [
        q(f"q{i}", f"unique question {rng.random()} about topic {i} {rng.random()}")
        for i in range(80)
    ]
    a = assign_splits(qs)
    shuffled = qs[:]
    rng.shuffle(shuffled)
    assert assign_splits(shuffled) == a


def test_adding_queries_never_moves_existing_ones() -> None:
    base = [
        q(f"q{i}", f"distinct question {i} regarding thing{i} alpha{i} beta{i}") for i in range(50)
    ]
    before = assign_splits(base)
    extra = [
        q(f"n{i}", f"another novel question {i} concerning item{i} gamma{i} delta{i}")
        for i in range(20)
    ]
    after = assign_splits(base + extra)
    assert all(after[k] == v for k, v in before.items())


def test_bucket_proportions_are_close_to_60_20_20() -> None:
    from collections import Counter

    counts = Counter(split_for_group_key(f"key-{i}") for i in range(5000))
    assert abs(counts["train"] / 5000 - 0.6) < 0.03
    assert abs(counts["val"] / 5000 - 0.2) < 0.03
    assert abs(counts["test"] / 5000 - 0.2) < 0.03


def test_leakage_check_flags_near_duplicates_across_splits(corpus: Corpus) -> None:
    qs = [
        q("q1", "read a file line by line python", "train"),
        q("q2", "python read a file line by line", "test"),
    ]
    report = check_dataset(qs, corpus)
    assert any("near-duplicate" in e for e in report.errors)


def test_leakage_check_flags_identical_text_and_split_drift_and_empty_split(corpus: Corpus) -> None:
    qs = [
        q("q1", "Exactly   the same text", "train"),
        q("q2", "exactly the same text", "train"),
        q("q3", "something else entirely different here", "train"),
    ]
    errors = check_dataset(qs, corpus).errors
    assert any("identical query text" in e for e in errors)
    assert any("empty split" in e for e in errors)


def test_leakage_check_accepts_a_clean_dataset_and_reports_stats(corpus: Corpus) -> None:
    qs = [
        q(f"q{i}", f"entirely separate question about subject{i} with words{i} unique{i}")
        for i in range(60)
    ]
    splits = assign_splits(qs)
    for item in qs:
        item.split = splits[item.id]
    report = check_dataset(qs, corpus)
    assert report.ok, report.errors
    assert report.stats["n_queries"] == 60
    assert sum(report.stats["by_split"].values()) == 60  # type: ignore[attr-defined]


def test_validate_queries_flags_unknown_sections_and_duplicates(corpus: Corpus) -> None:
    errors = validate_queries(
        [q("q1", "text one", rel=["python/1.0/page#nope"]), q("q1", "text two")], corpus
    )
    assert any("not in corpus" in e for e in errors)
    assert any("duplicate query id" in e for e in errors)


def test_authored_loader_expands_prefixes_merges_pooled_and_sets_provenance(tmp_path: Path) -> None:
    (tmp_path / "queries_x.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "pt001",
                    "query": "cuda out of memory",
                    "slice": "error_message",
                    "relevant": ["pt:notes/cuda#memory-management"],
                },
                {
                    "id": "ptfaq001",
                    "query": "My GPU memory isn't freed",
                    "slice": "official_faq",
                    "relevant": ["pt:notes/faq#x"],
                },
            ]
        )
    )
    (tmp_path / "pooled_judgments.yaml").write_text(
        yaml.safe_dump(
            {"pt001": {"added": ["pt:cuda#memory-management", "pt:notes/cuda#memory-management"]}}
        )
    )
    qs = {r.id: r for r in load_authored(tmp_path)}
    assert qs["pt001"].authored_relevant == ["pytorch/2.14/notes/cuda#memory-management"]
    assert qs["pt001"].relevant == [
        "pytorch/2.14/notes/cuda#memory-management",
        "pytorch/2.14/cuda#memory-management",
    ]
    assert qs["pt001"].framework == "pytorch"
    assert (qs["pt001"].label_source, qs["pt001"].verification) == (
        "claude_authored",
        "llm_authored_unreviewed",
    )
    assert (qs["ptfaq001"].label_source, qs["ptfaq001"].verification) == (
        "official_faq_heading",
        "official_source",
    )
