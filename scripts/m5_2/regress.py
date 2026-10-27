import json
from pathlib import Path

from neural_debugger.data.queries import load_materialised
from neural_debugger.ml.evaluation.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from neural_debugger.rag.coverage_eval import build_pipeline
from neural_debugger.rag.evaluation import load_items

items = load_items()
dev = [q for q in load_materialised(Path("data/eval/queries.jsonl")) if q.split in ("train", "val")]
out = {}
for name, (dirs, emb, prof) in {
    "base m5": (["data/corpus"], None, "m5"),
    "overlay coverage": (
        ["data/corpus", "data/corpus_supplement"],
        "data/embeddings_rag",
        "coverage",
    ),
}.items():
    pipe = build_pipeline(dirs, emb, "hybrid-gte-tuned", "cpu", prof)
    nd5 = []
    nd10 = []
    h5 = []
    mrr = []
    for rec in dev:
        ids = [s.section_id for s in pipe.retriever.search(rec.query, 100, None)]
        rel = set(rec.relevant)
        nd5.append(ndcg_at_k(ids, rel, 5))
        nd10.append(ndcg_at_k(ids, rel, 10))
        h5.append(hit_at_k(ids, rel, 5))
        mrr.append(reciprocal_rank(ids, rel))
    gate = {}
    for it in items:
        gate[it.id] = (it, pipe.prepare(it.query, None).sufficiency)
    tr = [s for it, s in gate.values() if it.group == "authored" and it.split == "train"]
    va = [s for it, s in gate.values() if it.group == "authored" and it.split == "val"]
    un = [(it, s) for it, s in gate.values() if it.group == "unanswerable"]
    res = {
        "NDCG@5": round(sum(nd5) / len(nd5), 4),
        "NDCG@10": round(sum(nd10) / len(nd10), 4),
        "hit@5": round(sum(h5) / len(h5), 4),
        "MRR": round(sum(mrr) / len(mrr), 4),
        "false_abstain_train": round(sum(s.level == "none" for s in tr) / len(tr), 4),
        "false_abstain_val": round(sum(s.level == "none" for s in va) / len(va), 4),
        "unanswerable_abstain_all": f"{sum(s.level == 'none' for _, s in un)}/{len(un)}",
        "unanswerable_abstain_calib": f"{sum(s.level == 'none' for it, s in un if it.split == 'calib')}/{sum(it.split == 'calib' for it, _ in un)}",
        "unanswerable_abstain_heldout_half": f"{sum(s.level == 'none' for it, s in un if it.split == 'heldout')}/{sum(it.split == 'heldout' for it, _ in un)}",
    }
    out[name] = res
    print(name, res)
Path("data/eval/results/m5_2/dev_regression.json").write_text(
    json.dumps(
        {
            "label": "development queries (train+val, n=%d) + unanswerable dev set; base corpus vs overlay"
            % len(dev),
            "results": out,
        },
        indent=1,
    )
)
