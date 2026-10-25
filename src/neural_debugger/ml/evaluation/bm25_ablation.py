"""BM25 configuration study on the *dev* queries only (train + val; test is never touched).

BM25 has no trainable parameters, so train and val are both legitimate development data; the test
split is reserved for the single final evaluation of the chosen configuration.

    python -m neural_debugger.ml.evaluation.bm25_ablation --out data/eval/results/bm25_ablation_dev.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.queries import load_materialised
from neural_debugger.ml.retrieval.bm25 import BM25Config, BM25Retriever
from neural_debugger.ml.retrieval.tokenizer import TokenizerConfig

from .evaluate import evaluate_queries, summarize_results
from .metrics import bootstrap_ci

DEV_SPLITS = ("train", "val")


def tokenizer_variants() -> dict[str, TokenizerConfig]:
    return {
        "plain words (baseline)": TokenizerConfig(mode="plain"),
        "plain + stopwords kept": TokenizerConfig(mode="plain", remove_stopwords=False),
        "identifiers (default)": TokenizerConfig(),
        "identifiers, parts only": TokenizerConfig(keep_compounds=False),
        "identifiers + stopwords kept": TokenizerConfig(remove_stopwords=False),
        "identifiers + S-stemmer": TokenizerConfig(stem=True),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/results/bm25_ablation_dev.json"))
    args = ap.parse_args(argv)

    corpus = load_corpus(args.corpus_dir)
    dev = [q for q in load_materialised(args.queries) if q.split in DEV_SPLITS]
    rows: list[dict[str, Any]] = []

    reference = [
        r.metrics["ndcg@10"]
        for r in evaluate_queries(BM25Retriever(corpus.chunks, BM25Config()), dev)
    ]

    def run(label: str, cfg: BM25Config) -> None:
        results = evaluate_queries(BM25Retriever(corpus.chunks, cfg), dev)
        summary = summarize_results(results)
        s = summary.as_dict()
        diffs = [r.metrics["ndcg@10"] - ref for r, ref in zip(results, reference, strict=True)]
        lo, hi = bootstrap_ci(diffs, seed=0)
        mean_diff = sum(diffs) / len(diffs)
        rows.append(
            {
                "label": label,
                "config": cfg.as_dict(),
                **s,
                "ndcg10_paired_diff_vs_default": {
                    "mean": round(mean_diff, 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                },
            }
        )
        m = summary.mean
        print(
            f"{label:30s} R@1 {m['recall@1']:.3f} R@10 {m['recall@10']:.3f} MRR {m['mrr']:.3f} "
            f"NDCG@10 {m['ndcg@10']:.3f} | vs default {mean_diff:+.3f} [{lo:+.3f},{hi:+.3f}]"
        )

    print(
        f"dev queries: {len(dev)} (splits {DEV_SPLITS}); test untouched\n-- tokenizer variants (k1=1.2, b=0.75)"
    )
    for label, tok in tokenizer_variants().items():
        run(label, BM25Config(tokenizer=tok))
    print("-- k1 x b grid (default tokenizer)")
    for k1, b in itertools.product((0.9, 1.2, 1.5, 2.0), (0.5, 0.75, 0.9)):
        run(f"k1={k1} b={b}", BM25Config(k1=k1, b=b))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"dev_splits": list(DEV_SPLITS), "n_dev_queries": len(dev), "runs": rows}, indent=1
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
