import time
from pathlib import Path

import yaml

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.queries import load_materialised
from neural_debugger.ml.evaluation.metrics import hit_at_k, ndcg_at_k
from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config
from neural_debugger.rag.question import prepare_question

corpus = load_corpus("data/corpus")
cfg = load_neural_config("configs/neural.yaml", "configs/m4.yaml")
fac = SystemFactory(corpus, cfg, device="cpu")
gold_items = [
    i
    for i in yaml.safe_load(open("data/eval/rag/coverage_dev.yaml"))["questions"]
    if i["gold"] and not any(g.get("supplement") for g in i["gold"])
]
dev = [q for q in load_materialised(Path("data/eval/queries.jsonl")) if q.split in ("train", "val")]
for name in [
    "hybrid-gte-tuned",
    "rerank-l6-n30",
    "rerank-l12-n30",
    "rerank-l12-n30-rrf",
    "rerank-l6-n30-rrf",
]:
    r = fac.build(name)
    ranks = []
    for it in gold_items:
        q = prepare_question(it["q"]).retrieval_query
        ids = [h.section_id for h in r.search(q, k=40, framework=None)]
        ranks.append(
            min([ids.index(g["section"]) + 1 for g in it["gold"] if g["section"] in ids] or [999])
        )
    nd = []
    h5 = []
    t = time.perf_counter()
    for rec in dev:
        ids = [h.section_id for h in r.search(rec.query, k=40, framework=None)]
        rel = set(rec.relevant)
        nd.append(ndcg_at_k(ids, rel, 5))
        h5.append(hit_at_k(ids, rel, 5))
    ms = (time.perf_counter() - t) * 1000 / len(dev)
    print(
        f"{name:20s} gold ranks {ranks} within5={sum(x <= 5 for x in ranks)} | dev NDCG@5={sum(nd) / len(nd):.3f} hit@5={sum(h5) / len(h5):.3f} | {ms:.0f} ms/query"
    )
