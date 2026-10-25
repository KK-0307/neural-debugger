"""Frozen-baseline manifest: hashes of everything later milestones must not silently change.

    python -m neural_debugger.ml.evaluation.freeze create    # writes data/eval/baseline_freeze.json
    python -m neural_debugger.ml.evaluation.freeze verify    # exit 1 on any drift

Frozen: corpus, labeled queries and splits, the review file, the BM25 configuration, and the BM25
result files. New work adds new files and new versions; it does not edit these.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from neural_debugger.data.queries import load_materialised
from neural_debugger.ml.retrieval.bm25 import BM25Config

FREEZE_PATH = Path("data/eval/baseline_freeze.json")
FROZEN_FILES = [
    "data/corpus/chunks.jsonl",
    "data/corpus/documents.jsonl",
    "data/eval/queries.jsonl",
    "data/eval/label_review.yaml",
    "data/eval/results/bm25_default.json",
    "data/eval/results/bm25_default_fwfilter.json",
    "data/eval/results/bm25_default_per_query.jsonl",
    "data/eval/results/bm25_ablation_dev.json",
    "data/eval/results/random_seed0.json",
    "data/manifests/corpus_manifest.json",
    "data/manifests/eval_dataset_manifest.json",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(root: Path = Path(".")) -> dict[str, Any]:
    queries = load_materialised(root / "data/eval/queries.jsonl")
    split_assignment = json.dumps({q.id: q.split for q in queries}, sort_keys=True)
    return {
        "baseline_id": "bm25-v1",
        "bm25_config": BM25Config().as_dict(),
        "split_counts": {s: sum(q.split == s for q in queries) for s in ("train", "val", "test")},
        "split_assignment_sha256": hashlib.sha256(split_assignment.encode()).hexdigest(),
        "files": {f: _sha(root / f) for f in FROZEN_FILES},
        "policy": (
            "The test split is used only for final evaluation of frozen configurations "
            "(see ml/evaluation/protocol.py and data/eval/results/test_access_log.jsonl)."
        ),
    }


def verify(root: Path = Path(".")) -> list[str]:
    frozen = json.loads((root / FREEZE_PATH).read_text())
    now = compute(root)
    problems = []
    for key in ("bm25_config", "split_counts", "split_assignment_sha256"):
        if frozen[key] != now[key]:
            problems.append(f"{key} changed")
    for f, h in frozen["files"].items():
        if now["files"].get(f) != h:
            problems.append(f"{f} changed")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["create", "verify"])
    args = ap.parse_args(argv)
    if args.command == "create":
        FREEZE_PATH.write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {FREEZE_PATH}")
        return 0
    problems = verify()
    for p in problems:
        print("DRIFT:", p)
    print("frozen baseline intact" if not problems else f"{len(problems)} drift(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
