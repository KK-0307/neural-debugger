"""Evaluate one named retrieval system (configs/neural.yaml) and write a JSON report.

    python -m neural_debugger.ml.evaluation.system_eval --system hybrid-bge-small --splits train val
    python -m neural_debugger.ml.evaluation.system_eval --system bm25 --qrels expanded --splits train val

Each system is meant to be run in its own process so peak memory is attributable. The held-out test
split needs ``--final-test`` (and is logged); development uses train + val.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.queries import QueryRecord, load_materialised
from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config

from .evaluate import QueryResult, hardware_info, latency_summary
from .metrics import query_metrics, summarize
from .protocol import TestSplitProtectedError, guard_test_access
from .qrels import Qrels, condense, load_qrels, unjudged_at_k

WARMUP_QUERIES = 3
RANK_DEPTH = 100
SAVED_RANK_DEPTH = 30


def current_rss_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(__import__("os").getpid())], capture_output=True, text=True
    )
    return float(out.stdout.strip() or 0) / 1024.0


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (
        peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    )  # bytes on macOS, KB on Linux


def flagged_query_ids(path: Path = Path("data/eval/label_review.yaml")) -> set[str]:
    if not path.exists():
        return set()
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {q for q, v in doc["verdicts"].items() if "uncertain" in v or "weak" in v}


def score_queries(
    retriever: Any, queries: list[QueryRecord], qrels: Qrels, framework_filter: bool = False
) -> tuple[list[QueryResult], dict[str, float], dict[str, dict[str, float]]]:
    for q in queries[:WARMUP_QUERIES]:  # lazy init / first-call costs are not query latency
        retriever.search(q.query, k=10, framework=None)
    results: list[QueryResult] = []
    unjudged: dict[str, float] = {}
    condensed: dict[str, dict[str, float]] = {}
    for q in queries:
        started = time.perf_counter()
        hits = retriever.search(
            q.query, k=RANK_DEPTH, framework=q.framework if framework_filter else None
        )
        latency = (time.perf_counter() - started) * 1000
        ranking = [h.section_id for h in hits]
        rel = qrels.relevant[q.id]
        results.append(QueryResult(q.id, ranking, query_metrics(ranking, rel), latency))
        unjudged[q.id] = unjudged_at_k(ranking, qrels.judged[q.id], 10)
        cond = query_metrics(condense(ranking, qrels.judged[q.id]), rel)
        condensed[q.id] = {
            k: v for k, v in cond.items() if k in {"mrr", "ndcg@10", "recall@10", "recall@1"}
        }
    return results, unjudged, condensed


def group_report(
    ids: list[str],
    results: dict[str, QueryResult],
    unjudged: dict[str, float],
    condensed: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not ids:
        return {"n_queries": 0}
    s = summarize({i: results[i].metrics for i in ids}, seed=0).as_dict()
    s["unjudged@10"] = round(sum(unjudged[i] for i in ids) / len(ids), 4)
    s["condensed"] = {
        k: round(sum(condensed[i][k] for i in ids) / len(ids), 4)
        for k in ("recall@1", "recall@10", "mrr", "ndcg@10")
    }
    s["latency"] = latency_summary([results[i] for i in ids])
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/neural.yaml"))
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"]
    )
    ap.add_argument("--qrels", choices=["original", "expanded"], default="original")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--framework-filter", action="store_true", help="oracle: use each query's true framework"
    )
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rankings-out", type=Path, help="write top-30 sections per query (JSONL)")
    args = ap.parse_args(argv)
    try:
        guard_test_access(
            args.splits, args.final_test, purpose=f"system_eval:{args.system}:{args.qrels}"
        )
    except TestSplitProtectedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    corpus = load_corpus(args.corpus_dir)
    all_queries = load_materialised(args.queries)
    queries = [q for q in all_queries if q.split in args.splits]
    qrels = load_qrels(queries, corpus, args.qrels)
    cfg = load_neural_config(args.config)
    rss_before = current_rss_mb()
    factory = SystemFactory(corpus, cfg, device=args.device)
    started = time.perf_counter()
    retriever = factory.build(args.system)
    build_seconds = time.perf_counter() - started
    rss_after_build = current_rss_mb()

    results, unjudged, condensed = score_queries(retriever, queries, qrels, args.framework_filter)
    by_id = {r.query_id: r for r in results}
    flagged = flagged_query_ids()
    authored = [q.id for q in queries if q.slice != "official_faq"]
    groups: dict[str, list[str]] = {
        "authored": authored,
        "official_faq": [q.id for q in queries if q.slice == "official_faq"],
        "authored_excl_flagged": [i for i in authored if i not in flagged],
    }
    for sl in ("error_message", "question"):
        groups[f"authored/{sl}"] = [q.id for q in queries if q.slice == sl]
    for fw in ("python", "pytorch", "fastapi"):
        groups[f"authored/{fw}"] = [
            q.id for q in queries if q.slice != "official_faq" and q.framework == fw
        ]
    report = {
        "system": args.system,
        "system_spec": cfg.systems[args.system].model_dump(),
        "splits": args.splits,
        "qrels": args.qrels,
        "oracle_framework_filter": args.framework_filter,
        "corpus_chunks_sha256": corpus.sha256,
        "n_queries": len(queries),
        "hardware": hardware_info(),
        "build_seconds": round(build_seconds, 3),
        "rss_mb": {
            "before": round(rss_before, 1),
            "after_build": round(rss_after_build, 1),
            "peak": round(peak_rss_mb(), 1),
        },
        "embeddings": {
            m: {k: v[k] for k in ("embedding_seconds", "device", "bytes", "n_chunks")}
            | {"fingerprint": v["encoder"]}
            for m, v in factory.info.embedding.items()
        },
        "groups": {g: group_report(ids, by_id, unjudged, condensed) for g, ids in groups.items()},
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if args.rankings_out:
        args.rankings_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rankings_out.open("w", encoding="utf-8") as f:
            for q in queries:
                r = by_id[q.id]
                f.write(
                    json.dumps(
                        {
                            "id": q.id,
                            "system": args.system,
                            "ranking": r.ranking[:SAVED_RANK_DEPTH],
                            "relevant": sorted(qrels.relevant[q.id]),
                            **r.metrics,
                        }
                    )
                    + "\n"
                )
    g = report["groups"]
    for name in ("authored", "official_faq"):
        m = g[name]
        if m["n_queries"]:
            mm = m["mean"]
            print(
                f"{args.system:22s} {name:13s} n={m['n_queries']:3d} R@1 {mm['recall@1']:.3f} R@5 {mm['recall@5']:.3f} "
                f"R@10 {mm['recall@10']:.3f} MRR {mm['mrr']:.3f} NDCG@10 {mm['ndcg@10']:.3f} unjudged@10 {m['unjudged@10']:.2f}"
            )
    print(
        f"build {build_seconds:.2f}s  rss after build {rss_after_build:.0f} MB  peak {peak_rss_mb():.0f} MB  "
        f"latency p50 {g['authored']['latency']['p50_ms']} ms p95 {g['authored']['latency']['p95_ms']} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
