"""Render markdown tables from the evaluation JSON files (so docs never hand-copy numbers).

uv run python scripts/render_eval_tables.py > /tmp/tables.md
"""

import json
from pathlib import Path

RES = Path("data/eval/results")
COLS = [
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("mrr", "MRR"),
    ("ndcg@10", "NDCG@10"),
]


def cell(s: dict, key: str, ci: bool = True) -> str:  # type: ignore[type-arg]
    m = s["mean"][key]
    lo, hi = s["ci95"][key]
    return f"{m:.3f} [{lo:.2f}, {hi:.2f}]" if ci else f"{m:.3f}"


def table(rows: list[tuple[str, dict]], ci: bool = True) -> str:  # type: ignore[type-arg]
    out = ["| | n | " + " | ".join(h for _, h in COLS) + " |", "|---|---:|" + "---:|" * len(COLS)]
    for name, s in rows:
        out.append(
            f"| {name} | {s['n_queries']} | " + " | ".join(cell(s, k, ci) for k, _ in COLS) + " |"
        )
    return "\n".join(out)


bm25 = json.loads((RES / "bm25_default.json").read_text())
fw = json.loads((RES / "bm25_default_fwfilter.json").read_text())
rnd = json.loads((RES / "random_seed0.json").read_text())
abl = json.loads((RES / "bm25_ablation_dev.json").read_text())

print("### Headline: authored queries (error messages + questions), BM25 default\n")
print(table([(f"{k}", v) for k, v in bm25["by_split_authored_only"].items()]))
print("\n### All queries including the official-FAQ slice\n")
print(table([(f"{k}", v) for k, v in bm25["by_split"].items()]))
print("\n### By slice (all splits pooled)\n")
print(table([(k, v) for k, v in bm25["all_queries"]["by_slice"].items()]))
print("\n### By framework (all splits pooled)\n")
print(table([(k, v) for k, v in bm25["all_queries"]["by_framework"].items()]))
print("\n### Test split by slice\n")
print(table([(k, v) for k, v in bm25["test_by_slice"].items()]))
print("\n### BM25 vs BM25 + oracle framework filter vs random (test, authored only)\n")
print(
    table(
        [
            ("random (seed 0)", rnd["by_split_authored_only"]["test"]),
            ("BM25", bm25["by_split_authored_only"]["test"]),
            ("BM25 + framework filter", fw["by_split_authored_only"]["test"]),
        ],
        ci=False,
    )
)
print("\n### Latency and build time (this machine)\n")
lat = bm25["by_split"]["test"]["latency"]
print(f"- hardware: {bm25['hardware']}")
print(f"- index build: {bm25['index_build_seconds']} s for {bm25['corpus']['n_chunks']} chunks")
print(
    f"- per-query search (test, n={lat['n']}): mean {lat['mean_ms']} ms, p50 {lat['p50_ms']} ms, p95 {lat['p95_ms']} ms, max {lat['max_ms']} ms"
)
print(f"\n### Dev-set configuration study (train+val, n={abl['n_dev_queries']})\n")
print(
    "| configuration | R@1 | R@10 | MRR | NDCG@10 | paired ΔNDCG@10 vs default [95% CI] |\n|---|---:|---:|---:|---:|---:|"
)
for r in abl["runs"]:
    d = r["ndcg10_paired_diff_vs_default"]
    m = r["mean"]
    print(
        f"| {r['label']} | {m['recall@1']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} | {m['ndcg@10']:.3f} | {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |"
    )
