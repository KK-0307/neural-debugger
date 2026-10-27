"""Frozen-baseline manifest: hashes of everything later milestones must not silently change.

    python -m neural_debugger.ml.evaluation.freeze create    # writes data/eval/baseline_freeze.json
    python -m neural_debugger.ml.evaluation.freeze verify    # exit 1 on any drift
    python -m neural_debugger.ml.evaluation.freeze create-m3 / verify-m3   # Milestone 3 configuration

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


M3_FREEZE_PATH = Path("data/eval/m3_freeze.json")
M3_TEST_SYSTEMS = ["bm25", "dense-gte-small", "hybrid-gte-tuned", "rerank-l12-n30-rrf"]
M3_FROZEN_FILES = [
    "configs/neural.yaml",
    "data/embeddings/gte-small/embeddings.npy",
    "data/embeddings/gte-small/meta.json",
    "data/eval/pooled_judgments_neural.yaml",
    "data/eval/results/m3/fusion_tuning_dev.json",
    *[f"data/eval/results/m3/dev_final/{s}.json" for s in M3_TEST_SYSTEMS],
    *[f"data/eval/results/m3/dev_final/{s}.rankings.jsonl" for s in M3_TEST_SYSTEMS],
]


def compute_m3(root: Path = Path(".")) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load((root / "configs/neural.yaml").read_text())
    return {
        "freeze_id": "neural-retrieval-v1",
        "depends_on": "bm25-v1 (data/eval/baseline_freeze.json)",
        "test_systems": {s: cfg["systems"][s] for s in M3_TEST_SYSTEMS},
        "embedding_model": cfg["embedding_models"]["gte-small"],
        "reranker": cfg["rerankers"]["ce-minilm-l12"],
        "selection": (
            "Chosen on train+val only with expanded judgments: embedding model = best dense dev NDCG@10; "
            "fusion = (k=10, w_dense=1.5) per the declared paired-CI rule; reranker = best dev variant; "
            "router rejected (dev regression)."
        ),
        "files": {f: _sha(root / f) for f in M3_FROZEN_FILES},
    }


def verify_m3(root: Path = Path(".")) -> list[str]:
    frozen = json.loads((root / M3_FREEZE_PATH).read_text())
    now = compute_m3(root)
    problems = [
        k + " changed"
        for k in ("test_systems", "embedding_model", "reranker")
        if frozen[k] != now[k]
    ]
    problems += [f + " changed" for f, h in frozen["files"].items() if now["files"].get(f) != h]
    return problems


M4_FREEZE_PATH = Path("data/eval/m4_freeze.json")
M4_TEST_SYSTEMS = [
    "bm25",
    "hybrid-gte-tuned",
    "dense-custom-tiny",
    "hybrid-custom-tiny",
    "dense-custom-tiny-real",
    "hybrid-custom-tiny-real",
]
M4_FROZEN_FILES = [
    "configs/m4.yaml",
    "configs/train/tiny.yaml",
    "configs/train/tiny-real.yaml",
    *[
        f"models/{m}/{f}"
        for m in ("custom-tiny", "custom-tiny-real")
        for f in ("best.pt", "tokenizer.json", "config.json", "summary.json")
    ],
    "data/eval/pooled_judgments_m4.yaml",
    *[f"data/eval/results/m4/dev_m4/{s}.json" for s in M4_TEST_SYSTEMS],
    *[f"data/eval/results/m4/dev_m4/{s}.rankings.jsonl" for s in M4_TEST_SYSTEMS],
]


def compute_m4(root: Path = Path(".")) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load((root / "configs/m4.yaml").read_text())
    return {
        "freeze_id": "custom-transformer-v1",
        "depends_on": "neural-retrieval-v1 (data/eval/m3_freeze.json)",
        "test_systems": {
            s: cfg["systems"].get(s, "frozen in configs/neural.yaml") for s in M4_TEST_SYSTEMS
        },
        "custom_models": cfg["custom_models"],
        "selection": (
            "Architecture/size chosen on dev (train+val) NDCG@10 averaged over 3 seeds (tiny > main > base); "
            "negatives: in-batch (BM25 hard negatives were worse); shipped weights are seed 0 of each "
            "config, fixed before any evaluation of the shipped files. tiny-real (real train queries) is "
            "judged on val only: its train/dev numbers are in-sample."
        ),
        "files": {f: _sha(root / f) for f in M4_FROZEN_FILES},
    }


def verify_m4(root: Path = Path(".")) -> list[str]:
    frozen = json.loads((root / M4_FREEZE_PATH).read_text())
    now = compute_m4(root)
    problems = [k + " changed" for k in ("test_systems", "custom_models") if frozen[k] != now[k]]
    problems += [f + " changed" for f, h in frozen["files"].items() if now["files"].get(f) != h]
    return problems


M5_FREEZE_PATH = Path("data/eval/m5_freeze.json")
M5_FROZEN_FILES = [
    "data/eval/rag/unanswerable.yaml",
    "data/eval/results/m5/calibration_hybrid-gte-tuned.json",
    "data/eval/results/m5/rag_extractive_hybrid-gte-tuned.json",
    "data/eval/results/m5/rag_extractive_hybrid-custom-tiny.json",
    "data/eval/results/m5/rag_extractive_bm25.json",
]


def compute_m5(root: Path = Path(".")) -> dict[str, Any]:
    return {
        "freeze_id": "rag-v1",
        "depends_on": "custom-transformer-v1 (data/eval/m4_freeze.json)",
        "note": (
            "Dev-only RAG evaluation with the extractive test provider; abstention threshold 0.375 "
            "calibrated on train + half of the out-of-scope set; test split untouched; no live LLM run."
        ),
        "files": {f: _sha(root / f) for f in M5_FROZEN_FILES},
    }


def verify_m5(root: Path = Path(".")) -> list[str]:
    frozen = json.loads((root / M5_FREEZE_PATH).read_text())
    now = compute_m5(root)
    return [f + " changed" for f, h in frozen["files"].items() if now["files"].get(f) != h]


M51_FREEZE_PATH = Path("data/eval/m5_1_freeze.json")
M51_FROZEN_FILES = [
    "data/eval/rag/live_acceptance.yaml",
    "data/eval/rag/independent_gemini.yaml",
    "data/eval/rag/grounding_adversarial.jsonl",
    "data/eval/results/m5_1/semantic_grounding.json",
    "data/eval/results/m5_1/live_gemini_gemini-3.1-flash-lite.raw.json",
    "data/eval/results/m5_1/rerun_infra_errors/live_gemini_gemini-3.1-flash-lite.raw.json",
    "data/eval/results/m5_1/live_merged_as_run.json",
    "data/eval/results/m5_1/independent/live_gemini_gemini-3.1-flash-lite.raw.json",
    "data/eval/results/m5_1/independent/rerun_infra_errors/live_gemini_gemini-3.1-flash-lite.raw.json",
    "data/eval/results/m5_1/independent/live_merged_as_run.json",
]


def compute_m51(root: Path = Path(".")) -> dict[str, Any]:
    return {
        "freeze_id": "rag-live-and-grounding-v1",
        "depends_on": "rag-v1 (data/eval/m5_freeze.json)",
        "note": (
            "Recorded evidence of the Milestone 5.1 live Gemini free-tier runs (21 agent-authored and 30 "
            "model-generated questions; raw replies are the evidence and cannot be regenerated bit-for-bit) "
            "and the synthetic adversarial grounding set. Agent-authored / model-generated; no human labels."
        ),
        "files": {f: _sha(root / f) for f in M51_FROZEN_FILES},
    }


def verify_m51(root: Path = Path(".")) -> list[str]:
    frozen = json.loads((root / M51_FREEZE_PATH).read_text())
    now = compute_m51(root)
    return [f + " changed" for f, h in frozen["files"].items() if now["files"].get(f) != h]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "command",
        choices=[
            "create",
            "verify",
            "create-m3",
            "verify-m3",
            "create-m4",
            "verify-m4",
            "create-m5",
            "verify-m5",
            "create-m5_1",
            "verify-m5_1",
        ],
    )
    args = ap.parse_args(argv)
    if args.command == "create-m5_1":
        M51_FREEZE_PATH.write_text(json.dumps(compute_m51(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {M51_FREEZE_PATH}")
        return 0
    if args.command == "verify-m5_1":
        problems = verify_m51()
        for p_ in problems:
            print("DRIFT:", p_)
        print("frozen M5.1 evidence intact" if not problems else f"{len(problems)} drift(s)")
        return 1 if problems else 0
    if args.command == "create-m5":
        M5_FREEZE_PATH.write_text(json.dumps(compute_m5(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {M5_FREEZE_PATH}")
        return 0
    if args.command == "verify-m5":
        problems = verify_m5()
        for p_ in problems:
            print("DRIFT:", p_)
        print("frozen M5 evaluation intact" if not problems else f"{len(problems)} drift(s)")
        return 1 if problems else 0
    if args.command == "create-m4":
        M4_FREEZE_PATH.write_text(json.dumps(compute_m4(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {M4_FREEZE_PATH}")
        return 0
    if args.command == "verify-m4":
        problems = verify_m4()
        for p in problems:
            print("DRIFT:", p)
        print("frozen M4 configuration intact" if not problems else f"{len(problems)} drift(s)")
        return 1 if problems else 0
    if args.command == "create-m3":
        M3_FREEZE_PATH.write_text(json.dumps(compute_m3(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {M3_FREEZE_PATH}")
        return 0
    if args.command == "verify-m3":
        problems = verify_m3()
        for p in problems:
            print("DRIFT:", p)
        print("frozen M3 configuration intact" if not problems else f"{len(problems)} drift(s)")
        return 1 if problems else 0
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
