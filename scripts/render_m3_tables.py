"""Markdown tables for Milestone 3 from data/eval/results/m3/<dir>/*.json (never hand-copied).

uv run python scripts/render_m3_tables.py dev_final
"""

import json
import sys
from pathlib import Path

d = Path("data/eval/results/m3") / sys.argv[1]
SYSTEMS = sys.argv[2:] or [
    "bm25",
    "dense-minilm-l6",
    "dense-bge-small",
    "dense-gte-small",
    "hybrid-gte-small",
    "hybrid-gte-tuned",
    "rerank-l6-n30-rrf",
    "rerank-l12-n30-rrf",
    "rerank-l12-n30",
]


def load(name: str, suffix: str = "") -> dict | None:  # type: ignore[type-arg]
    p = d / f"{name}{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def metrics_table(group: str) -> str:
    out = [
        "| system | n | R@1 | R@5 | R@10 | MRR | NDCG@10 | unjudged@10 | condensed NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SYSTEMS:
        r = load(s)
        if not r:
            continue
        g = r["groups"][group]
        m = g["mean"]
        ci = g["ci95"]["ndcg@10"]
        out.append(
            f"| {s} | {g['n_queries']} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}] | {g['unjudged@10']:.2f} | {g['condensed']['ndcg@10']:.3f} |"
        )
    return "\n".join(out)


def cost_table() -> str:
    out = [
        "| system | device | build / load (s) | embed (s, one-off) | p50 / p95 query (ms) | RSS after build (MB) | peak RSS (MB) | extra storage (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SYSTEMS:
        for suffix, dev in (("", "auto"), (".cpu", "cpu")):
            r = load(s, suffix)
            if not r:
                continue
            emb = r["embeddings"]
            embed_s = sum(v["embedding_seconds"] for v in emb.values()) if emb else 0
            mb = sum(v["bytes"] for v in emb.values()) / 1e6 if emb else 0
            lat = r["groups"]["authored"]["latency"]
            used = ",".join(sorted({v["device"] for v in emb.values()})) if emb else "-"
            label = (
                ("mps" if used == "mps" and dev == "auto" else dev)
                if dev == "cpu" or emb
                else "cpu"
            )
            out.append(
                f"| {s} | {label} | {r['build_seconds']:.1f} | {embed_s:.1f} | {lat['p50_ms']:.1f} / {lat['p95_ms']:.1f} | "
                f"{r['rss_mb']['after_build']:.0f} | {r['rss_mb']['peak']:.0f} | {mb:.1f} |"
            )
    return "\n".join(out)


if __name__ == "__main__":
    print("### Authored queries\n\n" + metrics_table("authored"))
    print("\n### Verbatim FAQ queries\n\n" + metrics_table("official_faq"))
    print("\n### Cost\n\n" + cost_table())
