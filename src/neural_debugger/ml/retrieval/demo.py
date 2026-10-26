"""Command-line demo: BM25 search over the real ingested documentation.

python -m neural_debugger.ml.retrieval.demo "RuntimeError: Expected all tensors to be on the same device"
python -m neural_debugger.ml.retrieval.demo "how do I raise a 404" --framework fastapi -k 5
python -m neural_debugger.ml.retrieval.demo --json "KeyError"
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

from neural_debugger.data.corpus import load_corpus

from .bm25 import BM25Config, BM25Retriever
from .types import Retriever

EXAMPLES = [
    "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu",
    "FastAPI how to return a 404 error to the client",
    "UnboundLocalError referenced before assignment",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("query", nargs="*", help="query text (omit to run built-in examples)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--framework", choices=["python", "pytorch", "fastapi"])
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--system",
        default="bm25",
        help="system from configs/neural.yaml (default: bm25, no extras needed)",
    )
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda (neural systems)")
    args = ap.parse_args(argv)

    corpus = load_corpus(args.corpus_dir)
    if args.system == "bm25":
        retriever: Retriever = BM25Retriever(corpus.chunks, BM25Config())
    else:
        from .systems import SystemFactory, load_neural_config

        retriever = SystemFactory(corpus, load_neural_config(), device=args.device).build(
            args.system
        )
    queries = [" ".join(args.query)] if args.query else EXAMPLES
    results = []
    for q in queries:
        started = time.perf_counter()
        hits = retriever.search(q, k=args.k, framework=args.framework)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if args.json:
            results.append(
                {
                    "query": q,
                    "latency_ms": round(elapsed_ms, 2),
                    "hits": [h.as_dict() for h in hits],
                }
            )
            continue
        print(
            f"\nQuery: {q}\n{len(hits)} sections in {elapsed_ms:.1f} ms "
            f"({args.system} over {len(corpus.chunks)} chunks, corpus {corpus.sha256[:12]})"
        )
        for h in hits:
            print(
                f"\n  {h.rank}. [{h.framework} {h.version}] {' > '.join(h.heading_path[-2:])}   (score {h.score:.2f})"
            )
            print(f"     {h.url}")
            print(textwrap.indent(textwrap.fill(h.snippet, 96), "     "))
    if args.json:
        print(json.dumps(results, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
