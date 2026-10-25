"""List unlabeled BM25 top-k sections per query, for relevance review ("pooling").

    uv run python scripts/pool_candidates.py fa [--depth 8]

Prints candidates only; judgments are recorded by hand in data/eval/pooled_judgments.yaml.
Pooling from a single system biases the judged set toward what BM25 retrieves; see docs/evaluation.md.
"""

import argparse
from pathlib import Path

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.queries import load_authored
from neural_debugger.ml.retrieval.bm25 import BM25Retriever

ap = argparse.ArgumentParser()
ap.add_argument("prefix", help="query id prefix, e.g. fa / pt / py")
ap.add_argument("--depth", type=int, default=8)
args = ap.parse_args()

corpus = load_corpus(Path("data/corpus"))
bm25 = BM25Retriever(corpus.chunks)
for q in load_authored(Path("data/eval")):
    if not q.id.startswith(args.prefix) or q.slice == "official_faq":
        continue
    known = set(q.relevant)
    print(f"\n[{q.id}] {q.query[:110]}")
    shown = 0
    for h in bm25.search(q.query, k=args.depth, framework=q.framework or None):
        if h.section_id in known:
            continue
        short = h.section_id.split("/", 2)[2]
        body = " ".join(corpus.section_text(h.section_id).split())
        print(f"  r{h.rank} {short[:78]} :: {body[:110]}")
        shown += 1
