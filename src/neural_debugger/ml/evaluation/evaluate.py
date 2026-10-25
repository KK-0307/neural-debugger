"""Standalone retrieval evaluation.

    python -m neural_debugger.ml.evaluation.evaluate --retriever bm25
    python -m neural_debugger.ml.evaluation.evaluate --retriever bm25 --framework-filter
    python -m neural_debugger.ml.evaluation.evaluate --retriever random --seed 0

Retrieval is scored at *section* level (see metrics.py) against the labels in
``data/eval/queries.jsonl``. Every run writes a JSON file recording the config, corpus/dataset
hashes, hardware, latency and metrics so results are reproducible and attributable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_debugger.data.corpus import Corpus, load_corpus
from neural_debugger.data.queries import QueryRecord, load_materialised
from neural_debugger.data.schema import sha256_bytes
from neural_debugger.ml.retrieval.bm25 import BM25Config, BM25Retriever
from neural_debugger.ml.retrieval.random_baseline import RandomRetriever
from neural_debugger.ml.retrieval.tokenizer import TokenizerConfig
from neural_debugger.ml.retrieval.types import Retriever

from .metrics import METRIC_NAMES, MetricSummary, query_metrics, summarize
from .protocol import TestSplitProtectedError, guard_test_access

RETRIEVAL_DEPTH = 100  # sections retrieved per query; MRR is computed over this ranking
BOOTSTRAP_SEED = 0
SPLITS = ("train", "val", "test")


@dataclass
class QueryResult:
    query_id: str
    ranking: list[str]
    metrics: dict[str, float]
    latency_ms: float


def evaluate_queries(
    retriever: Retriever,
    queries: Sequence[QueryRecord],
    depth: int = RETRIEVAL_DEPTH,
    framework_filter: bool = False,
) -> list[QueryResult]:
    results = []
    for q in queries:
        started = time.perf_counter()
        hits = retriever.search(
            q.query, k=depth, framework=q.framework if framework_filter else None
        )
        latency_ms = (time.perf_counter() - started) * 1000
        ranking = [h.section_id for h in hits]
        results.append(
            QueryResult(q.id, ranking, query_metrics(ranking, set(q.relevant)), latency_ms)
        )
    return results


def summarize_results(results: Sequence[QueryResult]) -> MetricSummary:
    return summarize({r.query_id: r.metrics for r in results}, seed=BOOTSTRAP_SEED)


def latency_summary(results: Sequence[QueryResult]) -> dict[str, float]:
    ms = sorted(r.latency_ms for r in results)
    if not ms:
        return {}
    return {
        "n": len(ms),
        "mean_ms": round(statistics.mean(ms), 3),
        "p50_ms": round(ms[len(ms) // 2], 3),
        "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 3),
        "max_ms": round(ms[-1], 3),
    }


def hardware_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    if platform.system() == "Darwin":
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
    return info


def build_report(
    name: str,
    config: dict[str, Any],
    corpus: Corpus,
    queries: list[QueryRecord],
    results_by_id: dict[str, QueryResult],
    queries_path: Path,
    index_build_seconds: float | None,
    splits: Sequence[str] = SPLITS,
) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in splits:
        rs = [results_by_id[q.id] for q in queries if q.split == split]
        by_split[split] = {**summarize_results(rs).as_dict(), "latency": latency_summary(rs)}
    # Headline view: the official-FAQ slice is lexically trivial (the query is the indexed heading),
    # so the number to quote is the authored queries only.
    by_split_authored: dict[str, Any] = {}
    for split in splits:
        rs = [
            results_by_id[q.id] for q in queries if q.split == split and q.slice != "official_faq"
        ]
        by_split_authored[split] = summarize_results(rs).as_dict()
    breakdown: dict[str, dict[str, Any]] = {"by_slice": {}, "by_framework": {}}
    for key, attr in (("by_slice", "slice"), ("by_framework", "framework")):
        groups: dict[str, list[QueryResult]] = defaultdict(list)
        for q in queries:
            groups[getattr(q, attr)].append(results_by_id[q.id])
        breakdown[key] = {g: summarize_results(rs).as_dict() for g, rs in sorted(groups.items())}
    # test split only, by slice: what headline numbers hide
    test_by_slice: dict[str, list[QueryResult]] = defaultdict(list)
    for q in queries:
        if q.split == "test":
            test_by_slice[q.slice].append(results_by_id[q.id])
    return {
        "name": name,
        "config": config,
        "corpus": {
            "id": "nd-corpus-v1",
            "chunks_sha256": corpus.sha256,
            "n_chunks": len(corpus.chunks),
        },
        "dataset": {
            "queries_sha256": sha256_bytes(queries_path.read_bytes()),
            "n_queries": len(queries),
        },
        "retrieval_depth_sections": RETRIEVAL_DEPTH,
        "bootstrap": {"resamples": 2000, "seed": BOOTSTRAP_SEED, "interval": "95% percentile"},
        "index_build_seconds": None
        if index_build_seconds is None
        else round(index_build_seconds, 3),
        "hardware": hardware_info(),
        "by_split": by_split,
        "by_split_authored_only": by_split_authored,
        "all_queries": {**breakdown},
        "test_by_slice": {
            g: summarize_results(rs).as_dict() for g, rs in sorted(test_by_slice.items())
        },
        "metric_names": list(METRIC_NAMES),
    }


def format_table(report: dict[str, Any]) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
    lines = [f"{'split':24s} {'n':>4s} " + " ".join(f"{c:>10s}" for c in cols)]
    for key, label in (("by_split_authored_only", "authored"), ("by_split", "all+FAQ")):
        for split, s in report[key].items():
            name = f"{split} [{label}]"
            nums = " ".join(f"{s['mean'][c]:10.3f}" for c in cols)
            lines.append(f"{name:24s} {s['n_queries']:4d} {nums}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--retriever", choices=["bm25", "random"], default="bm25")
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    ap.add_argument("--per-query-out", type=Path, help="write per-query rankings/metrics (JSONL)")
    ap.add_argument(
        "--framework-filter", action="store_true", help="restrict search to the query's framework"
    )
    ap.add_argument("--k1", type=float, default=BM25Config().k1)
    ap.add_argument("--b", type=float, default=BM25Config().b)
    ap.add_argument("--tokenizer-mode", choices=["plain", "identifiers"], default="identifiers")
    ap.add_argument("--no-compounds", action="store_true")
    ap.add_argument("--keep-stopwords", action="store_true")
    ap.add_argument("--stem", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="random retriever seed")
    ap.add_argument(
        "--final-test",
        action="store_true",
        help="allow scoring the held-out test split (final evaluation of a frozen config only)",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["train", "val"],
        help="splits to score (default: train val; 'test' also needs --final-test)",
    )
    args = ap.parse_args(argv)

    try:
        guard_test_access(args.splits, args.final_test, purpose=f"evaluate:{args.retriever}")
    except TestSplitProtectedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    corpus = load_corpus(args.corpus_dir)
    queries = load_materialised(args.queries)
    retriever: Retriever
    build_seconds: float | None = None
    if args.retriever == "bm25":
        cfg = BM25Config(
            k1=args.k1,
            b=args.b,
            tokenizer=TokenizerConfig(
                mode=args.tokenizer_mode,
                keep_compounds=not args.no_compounds,
                remove_stopwords=not args.keep_stopwords,
                stem=args.stem,
            ),
        )
        bm25 = BM25Retriever(corpus.chunks, cfg)
        build_seconds = bm25.build_seconds
        retriever = bm25
        config: dict[str, Any] = {"retriever": "bm25", **cfg.as_dict()}
    else:
        retriever = RandomRetriever(corpus.chunks, seed=args.seed)
        config = {"retriever": "random", "seed": args.seed}
    config["framework_filter"] = args.framework_filter

    queries = [q for q in queries if q.split in args.splits]
    results = evaluate_queries(retriever, queries, framework_filter=args.framework_filter)
    by_id = {r.query_id: r for r in results}
    report = build_report(
        f"{args.retriever}{'+fwfilter' if args.framework_filter else ''}",
        config,
        corpus,
        queries,
        by_id,
        args.queries,
        build_seconds,
    )
    print(json.dumps(config))
    print(format_table(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_query_out:
        args.per_query_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_query_out.open("w", encoding="utf-8") as f:
            for q in queries:
                r = by_id[q.id]
                f.write(
                    json.dumps(
                        {
                            "id": q.id,
                            "split": q.split,
                            "slice": q.slice,
                            "framework": q.framework,
                            "relevant": q.relevant,
                            "top10": r.ranking[:10],
                            **r.metrics,
                        }
                    )
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
