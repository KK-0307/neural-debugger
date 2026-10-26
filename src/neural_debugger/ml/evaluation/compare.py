"""Paired comparison of systems from per-query ranking files (written by ``system_eval --rankings-out``).

    python -m neural_debugger.ml.evaluation.compare --dir data/eval/results/m3/dev_expanded \
        --reference bm25 --systems dense-gte-small hybrid-gte-small --group authored

Prints, for each system, the mean metric and the *paired* difference vs the reference with a 95%
bootstrap interval over queries, plus win/loss/tie counts on NDCG@10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .metrics import bootstrap_ci

METRICS = ("recall@1", "recall@10", "mrr", "ndcg@10")


def load_rankings(path: Path, group: str) -> dict[str, dict[str, float]]:
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def paired(ref: dict[str, dict[str, float]], other: dict[str, dict[str, float]], metric: str):  # type: ignore[no-untyped-def]
    ids = sorted(set(ref) & set(other))
    diffs = [other[i][metric] - ref[i][metric] for i in ids]
    lo, hi = bootstrap_ci(diffs, seed=0)
    wins = sum(d > 1e-9 for d in diffs)
    losses = sum(d < -1e-9 for d in diffs)
    return float(np.mean(diffs)), lo, hi, wins, losses, len(ids) - wins - losses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--group", choices=["authored", "faq", "all"], default="authored")
    ap.add_argument("--suffix", default=".rankings.jsonl")
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    args = ap.parse_args(argv)
    slices = {
        json.loads(line)["id"]: json.loads(line)["slice"]
        for line in args.queries.read_text().splitlines()
    }

    def keep(i: str) -> bool:
        return bool(args.group == "all" or (slices[i] == "official_faq") == (args.group == "faq"))

    def load(name: str) -> dict[str, dict[str, float]]:
        rows = load_rankings(args.dir / f"{name}{args.suffix}", args.group)
        return {i: r for i, r in rows.items() if keep(i)}

    ref = load(args.reference)
    print(
        f"reference {args.reference}: n={len(ref)} "
        + " ".join(f"{m} {np.mean([r[m] for r in ref.values()]):.3f}" for m in METRICS)
    )
    for s in args.systems:
        cur = load(s)
        means = " ".join(f"{m} {np.mean([r[m] for r in cur.values()]):.3f}" for m in METRICS)
        d, lo, hi, w, loss, t = paired(ref, cur, "ndcg@10")
        d1, lo1, hi1, *_ = paired(ref, cur, "recall@10")
        print(
            f"{s:20s} {means} | dNDCG@10 {d:+.3f} [{lo:+.3f},{hi:+.3f}] dR@10 {d1:+.3f} [{lo1:+.3f},{hi1:+.3f}] W/L/T {w}/{loss}/{t}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
