"""Evaluation protocol guards: the held-out test split is opt-in and every access is logged.

Development (model selection, hyper-parameter tuning, ablations) uses ``train`` and ``val`` only.
Scoring the ``test`` split requires an explicit ``--final-test`` flag and appends a line to
``data/eval/results/test_access_log.jsonl`` (who/what/when/config hash), so the history of test
access can be audited and a "touched too often" problem cannot hide.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

TEST_LOG = Path("data/eval/results/test_access_log.jsonl")
DEV_SPLITS = ("train", "val")


class TestSplitProtectedError(RuntimeError):
    __test__ = False  # not a pytest class


def guard_test_access(
    splits: Sequence[str],
    allow_test: bool,
    purpose: str,
    config: dict[str, object] | None = None,
    log_path: Path = TEST_LOG,
) -> None:
    """Raise unless ``test`` is absent or explicitly allowed; log allowed accesses."""
    if "test" not in splits:
        return
    if not allow_test:
        raise TestSplitProtectedError(
            "the held-out test split is protected: use --splits train val for development; "
            "pass --final-test only for the final evaluation of a frozen configuration"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": purpose,
        "argv": sys.argv[1:],
        "config": config or {},
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
