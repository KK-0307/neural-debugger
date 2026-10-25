"""Build and check the materialised evaluation dataset.

python -m neural_debugger.data.dataset build    # merge YAML sources -> queries.jsonl + manifest
python -m neural_debugger.data.dataset check    # integrity + leakage checks on queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .corpus import load_corpus
from .leakage import check_dataset
from .queries import QueryRecord, load_authored, load_materialised
from .schema import sha256_bytes
from .splits import assign_splits

EVAL_DIR = Path("data/eval")
MANIFEST = Path("data/manifests/eval_dataset_manifest.json")
CORPUS_MANIFEST = Path("data/manifests/corpus_manifest.json")


def materialise(queries: list[QueryRecord]) -> bytes:
    splits = assign_splits(queries)
    rows = []
    for q in sorted(queries, key=lambda q: q.id):
        q.split = splits[q.id]
        row = q.model_dump(exclude_none=True)
        row["relevant"] = q.relevant
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return ("\n".join(rows) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["build", "check"])
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args(argv)
    corpus = load_corpus(args.corpus_dir)
    out_path = args.eval_dir / "queries.jsonl"

    if args.command == "build":
        authored = load_authored(args.eval_dir)
        data = materialise(authored)
        out_path.write_bytes(data)

    queries = load_materialised(out_path)
    report = check_dataset(queries, corpus)
    for e in report.errors:
        print("ERROR:", e)

    if args.command == "build":
        corpus_manifest = (
            json.loads(CORPUS_MANIFEST.read_text()) if CORPUS_MANIFEST.exists() else {}
        )
        by_provenance = Counter((q.label_source, q.verification) for q in queries)
        manifest = {
            "dataset_id": "nd-eval-v1",
            "queries_sha256": sha256_bytes(out_path.read_bytes()),
            "corpus_id": corpus_manifest.get("corpus_id"),
            "corpus_chunks_sha256": corpus.sha256,
            "split_rule": "see neural_debugger/data/splits.py (salted sha256 buckets over near-duplicate groups)",
            "label_provenance": {f"{a} / {b}": n for (a, b), n in sorted(by_provenance.items())},
            "n_pooled_additions": sum(len(q.pooled_added) for q in queries),
            "human_reviewed_queries": 0,
            **report.stats,
        }
        args.manifest.write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report.stats, indent=1))
    print("OK" if report.ok else f"{len(report.errors)} problem(s)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
