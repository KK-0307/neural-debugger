"""Summarise training runs (mean +- std over seeds) from ``runs/<experiment>/<name>/seed*/summary.json``.

python -m neural_debugger.ml.training.aggregate runs/m4 [--out docs/results/m4_training.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KEYS = ("synth_mrr", "dev_ndcg@10", "dev_recall@10", "dev_mrr", "val_ndcg@10", "train_loss")


def collect(root: Path) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    for f in sorted(root.glob("*/seed*/summary.json")):
        s = json.loads(f.read_text())
        row = {k: float(s["best_epoch_metrics"][k]) for k in KEYS if k in s["best_epoch_metrics"]}
        row.update(
            seed=float(s["seed"]),
            best_epoch=float(s["best_epoch"]),
            params=float(s["n_parameters"]),
            seconds=float(s["train_seconds"]),
        )
        out.setdefault(f.parent.parent.name, []).append(row)
    return out


def ms(vals: list[float]) -> str:
    sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
    return f"{statistics.mean(vals):.3f} ± {sd:.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def _col(rows: list[dict[str, float]], k: str) -> list[float]:
    return [r[k] for r in rows if k in r]


def table(runs: dict[str, list[dict[str, float]]]) -> str:
    head = "| config | seeds | params | best epoch | synth MRR | dev NDCG@10 | dev R@10 | val NDCG@10 | train s |"
    lines = [head, "|" + "---|" * 9]
    for name, rows in runs.items():
        lines.append(
            f"| {name} | {len(rows)} | {int(rows[0]['params']):,} | {statistics.mean(_col(rows, 'best_epoch')):.1f} "
            f"| {ms(_col(rows, 'synth_mrr'))} | {ms(_col(rows, 'dev_ndcg@10'))} | {ms(_col(rows, 'dev_recall@10'))} "
            f"| {ms(_col(rows, 'val_ndcg@10'))} | {statistics.mean(_col(rows, 'seconds')):.0f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    text = table(collect(args.root))
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
