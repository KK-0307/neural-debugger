"""Markdown tables for Milestone 4, computed from per-query ranking files (never hand-copied).

    uv run python scripts/render_m4_tables.py data/eval/results/m4/dev_m4 --splits val
    uv run python scripts/render_m4_tables.py data/eval/results/m4/dev_m4 --splits train val --systems bm25 ...

Each row is the mean over the queries of the chosen splits/slice; the last columns are the paired
difference in NDCG@10 against two references (95% bootstrap interval over queries).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from neural_debugger.ml.evaluation.metrics import bootstrap_ci

DEFAULT = [
    "bm25",
    "dense-gte-small",
    "hybrid-gte-small",
    "hybrid-gte-tuned",
    "dense-custom-tiny",
    "hybrid-custom-tiny",
    "dense-custom-tiny-real",
    "hybrid-custom-tiny-real",
]


def load(d: Path, name: str) -> dict[str, dict]:  # type: ignore[type-arg]
    p = d / f"{name}.rankings.jsonl"
    return {json.loads(line)["id"]: json.loads(line) for line in p.read_text().splitlines()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--slice", choices=["authored", "faq"], default="authored")
    ap.add_argument("--systems", nargs="+", default=DEFAULT)
    ap.add_argument("--refs", nargs=2, default=["bm25", "hybrid-gte-tuned"])
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    a = ap.parse_args()
    meta = {
        r["id"]: r
        for r in map(json.loads, a.queries.read_text().splitlines())
        if r["split"] in a.splits
    }
    keep = {i for i, r in meta.items() if (r["slice"] == "official_faq") == (a.slice == "faq")}
    rows = {s: load(a.dir, s) for s in a.systems if (a.dir / f"{s}.rankings.jsonl").exists()}
    ids = sorted(keep & set.intersection(*[set(r) for r in rows.values()]))
    refs = {n: rows[n] for n in a.refs if n in rows}
    head = "| system | n | R@1 | R@10 | MRR | NDCG@10 |" + "".join(
        f" Δ NDCG@10 vs {n} |" for n in refs
    )
    print(head)
    print("|---|---:|---:|---:|---:|---:|" + "---:|" * len(refs))
    for s, r in rows.items():
        m = {
            k: float(np.mean([r[i][k] for i in ids]))
            for k in ("recall@1", "recall@10", "mrr", "ndcg@10")
        }
        cells = ""
        for n, ref in refs.items():
            if n == s:
                cells += " – |"
                continue
            diffs = [r[i]["ndcg@10"] - ref[i]["ndcg@10"] for i in ids]
            lo, hi = bootstrap_ci(diffs, seed=0)
            cells += f" {np.mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}] |"
        print(
            f"| {s} | {len(ids)} | {m['recall@1']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} | {m['ndcg@10']:.3f} |{cells}"
        )


if __name__ == "__main__":
    main()
