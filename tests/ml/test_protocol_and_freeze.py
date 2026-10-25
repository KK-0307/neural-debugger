import json
from pathlib import Path

import pytest
import yaml

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.queries import expand_section_id, load_materialised
from neural_debugger.ml.evaluation import evaluate, freeze
from neural_debugger.ml.evaluation.protocol import TestSplitProtectedError, guard_test_access

ROOT = Path(__file__).resolve().parents[2]


def test_test_split_is_blocked_by_default_and_dev_splits_are_free(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    guard_test_access(["train", "val"], False, "dev", log_path=log)
    assert not log.exists()
    with pytest.raises(TestSplitProtectedError):
        guard_test_access(["train", "test"], False, "oops", log_path=log)
    assert not log.exists()


def test_allowed_test_access_is_logged(tmp_path: Path) -> None:
    log = tmp_path / "sub" / "log.jsonl"
    guard_test_access(["test"], True, "final:frozen-config", config={"x": 1}, log_path=log)
    guard_test_access(["val", "test"], True, "final:again", log_path=log)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["purpose"] for r in rows] == ["final:frozen-config", "final:again"]
    assert rows[0]["config"] == {"x": 1} and rows[0]["at"]


def test_evaluate_cli_refuses_test_without_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert evaluate.main(["--retriever", "random", "--splits", "test"]) == 2
    assert "protected" in capsys.readouterr().err


@pytest.mark.skipif(not (ROOT / "data/eval/baseline_freeze.json").exists(), reason="no freeze file")
def test_frozen_baseline_is_intact() -> None:
    assert freeze.verify(ROOT) == []


@pytest.mark.skipif(not (ROOT / "data/eval/label_review.yaml").exists(), reason="no review file")
def test_label_review_is_valid_and_never_claims_human_verification() -> None:
    doc = yaml.safe_load((ROOT / "data/eval/label_review.yaml").read_text())
    assert doc["human_verified"] is False and doc["independent"] is False
    assert "agent" in doc["reviewer"] and "human" not in doc["reviewer"]
    queries = {q.id: q for q in load_materialised(ROOT / "data/eval/queries.jsonl")}
    sections = load_corpus(ROOT / "data/corpus").section_ids
    n = 0
    for qid, verdicts in doc["verdicts"].items():
        assert queries[qid].split in ("train", "val"), "review scope is dev splits only"
        relevant = set(queries[qid].relevant)
        for verdict, shorts in verdicts.items():
            assert verdict in doc["verdict_meaning"]
            for short in shorts:
                full = expand_section_id(short)
                assert full in sections and full in relevant, (qid, full)
                n += 1
    assert n == doc["n_judgments_reviewed"] >= 30
