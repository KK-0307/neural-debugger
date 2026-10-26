"""RRF parameter study on the dev queries (train+val, expanded judgments); test is never touched.

    python -m neural_debugger.ml.evaluation.tune_fusion --model gte-small

Component rankings (BM25 and dense) are computed once per query; every (k, weights) setting is then
scored from the cache. Selection rule (declared before looking): keep the default (k=60, weights 1:1)
unless another setting beats it on dev NDCG@10 *and* its paired 95% interval excludes zero.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.queries import load_materialised
from neural_debugger.ml.retrieval.fusion import fuse_hits
from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config

from .metrics import bootstrap_ci, query_metrics
from .qrels import load_qrels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="gte-small")
    ap.add_argument("--qrels", choices=["original", "expanded"], default="expanded")
    ap.add_argument("--out", type=Path, default=Path("data/eval/results/m3/fusion_tuning_dev.json"))
    args = ap.parse_args(argv)

    corpus = load_corpus(Path("data/corpus"))
    queries = [
        q
        for q in load_materialised(Path("data/eval/queries.jsonl"))
        if q.split in ("train", "val") and q.slice != "official_faq"
    ]
    qrels = load_qrels(queries, corpus, args.qrels)
    factory = SystemFactory(corpus, load_neural_config())
    bm25, dense = factory.build("bm25"), factory.build(f"dense-{args.model}")
    cache = {q.id: (bm25.search(q.query, 100), dense.search(q.query, 100)) for q in queries}

    def score(k: int, w_dense: float) -> list[float]:
        out = []
        for q in queries:
            fused = fuse_hits(list(cache[q.id]), [1.0, w_dense], k_rrf=k)
            out.append(
                query_metrics([h.section_id for h in fused[:100]], qrels.relevant[q.id])["ndcg@10"]
            )
        return out

    base = score(60, 1.0)
    rows = []
    print(
        f"dev authored n={len(queries)} qrels={args.qrels}; default (k=60, w_dense=1.0) NDCG@10 {sum(base) / len(base):.3f}"
    )
    for k, w in itertools.product((10, 30, 60, 100), (0.5, 1.0, 1.5, 2.0, 3.0)):
        s = score(k, w)
        d = [a - b for a, b in zip(s, base, strict=True)]
        lo, hi = bootstrap_ci(d, seed=0)
        rows.append(
            {
                "k_rrf": k,
                "w_dense": w,
                "ndcg@10": round(sum(s) / len(s), 4),
                "paired_diff_vs_default": [round(sum(d) / len(d), 4), round(lo, 4), round(hi, 4)],
            }
        )
        print(
            f"k={k:3d} w_dense={w:3.1f}  NDCG@10 {sum(s) / len(s):.3f}  vs default {sum(d) / len(d):+.3f} [{lo:+.3f},{hi:+.3f}]"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"model": args.model, "qrels": args.qrels, "n": len(queries), "rows": rows}, indent=1
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
