"""Does dense query-to-documentation cosine similarity separate answerable from out-of-scope questions better than lexical coverage?

Development data only. Reports AUC (probability a random answerable question scores above a random out-of-scope one) per signal.
"""

from __future__ import annotations

import yaml
from sklearn.metrics import roc_auc_score  # dev-only analysis tool (a dev dependency)

from neural_debugger.data.corpus import load_corpus_overlay
from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config
from neural_debugger.rag.coverage_eval import build_pipeline
from neural_debugger.rag.evaluation import load_items

corpus = load_corpus_overlay(["data/corpus", "data/corpus_supplement"])
from pathlib import Path

fac = SystemFactory(
    corpus,
    load_neural_config("configs/neural.yaml", "configs/m4.yaml"),
    device="cpu",
    embedding_dir=Path("data/embeddings_rag"),
)
dense = fac.build("dense-gte-small")
pipe = build_pipeline(
    ["data/corpus", "data/corpus_supplement"],
    "data/embeddings_rag",
    "hybrid-gte-tuned",
    "cpu",
    "coverage",
)
items = load_items()
vd = yaml.safe_load(open("data/eval/rag/verbose_dev.yaml"))["questions"]
cd = yaml.safe_load(open("data/eval/rag/coverage_dev.yaml"))["questions"]
tb = yaml.safe_load(open("data/eval/rag/traceback_dev.yaml"))["questions"]
pos = {
    "train_answerable": [i.query for i in items if i.group == "authored" and i.split == "train"],
    "val_answerable": [i.query for i in items if i.group == "authored" and i.split == "val"],
    "verbose_in_scope": [q["q"] for q in vd if q["kind"] == "answer"],
    "traceback_dev": [q["q"] for q in tb],
    "dev_in_scope": [q["q"] for q in cd if q["kind"] == "answer"],
}
neg = {
    "unanswerable": [i.query for i in items if i.group == "unanswerable"],
    "dev_oos": [q["q"] for q in cd if q["kind"] == "abstain"],
    "verbose_oos": [q["q"] for q in vd if q["kind"] == "abstain"],
}


def feats(text: str) -> dict[str, float]:
    p = pipe.prepare(text, None)
    q = p.question
    qv = dense.encode_query(q.retrieval_query)
    hits = dense.search_vector(qv, 5, None)
    scores = [h.score for h in hits]
    return {
        "coverage": p.sufficiency.coverage,
        "dense_top1": scores[0],
        "dense_mean5": sum(scores) / len(scores),
    }


F = {g: [feats(t) for t in ts] for g, ts in {**pos, **neg}.items()}
import sys

if len(sys.argv) > 1 and sys.argv[1] == "grid":
    # combined score = coverage + lam * (dense_top1 - 0.80) / 0.10 ; threshold from the Milestone 5 calibration rule
    tr = [f for f in F["train_answerable"]]
    cal = [f for g in ("unanswerable",) for f in F[g]]
    n_un = len(F["unanswerable"])
    # the unanswerable set is split calib/heldout by item; recover the split flags in the same order as load_items
    flags = [i.split for i in items if i.group == "unanswerable"]
    base = {"unans_calib": 13, "unans_heldout": 13, "dev_oos": 8, "verbose_oos": 9}
    print("M5 reference abstentions:", base, "| verbose in-scope abstain 18/36")
    results = []
    for lam in (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        sc = lambda f: f["coverage"] + lam * (f["dense_top1"] - 0.80) / 0.10
        trs = sorted(sc(f) for f in tr)
        calib = [sc(f) for f, fl in zip(F["unanswerable"], flags) if fl == "calib"]
        best = None
        lo, hi = min(trs) - 0.01, max(trs) + 0.01
        steps = 400
        for k in range(steps + 1):
            thr = lo + (hi - lo) * k / steps
            fa = sum(x < thr for x in trs) / len(trs)
            ab = sum(x < thr for x in calib) / len(calib)
            if fa <= 0.05 and (best is None or (ab, thr) >= (best[1], best[0])):
                best = (thr, ab, fa)
        thr = best[0]
        row = {
            "unans_calib": sum(
                sc(f) < thr for f, fl in zip(F["unanswerable"], flags) if fl == "calib"
            ),
            "unans_heldout": sum(
                sc(f) < thr for f, fl in zip(F["unanswerable"], flags) if fl == "heldout"
            ),
            "dev_oos": sum(sc(f) < thr for f in F["dev_oos"]),
            "verbose_oos": sum(sc(f) < thr for f in F["verbose_oos"]),
        }
        vin = sum(sc(f) < thr for f in F["verbose_in_scope"])
        tbd = sum(sc(f) < thr for f in F["traceback_dev"])
        din = sum(sc(f) < thr for f in F["dev_in_scope"])
        val = sum(sc(f) < thr for f in F["val_answerable"])
        ok = all(row[k] >= base[k] for k in base)
        print(
            f"lam={lam:<4} thr={thr:.3f} FA(train)={best[2]:.3f} val_FA={val}/35 | OOS {row} {'OK' if ok else 'worse'} | verbose in-scope abstain {vin}/36 | tb {tbd}/10 | dev in-scope {din}/39"
        )
    raise SystemExit

for name in ("coverage", "dense_top1", "dense_mean5"):
    for pg in ("train_answerable", "verbose_in_scope", "traceback_dev", "dev_in_scope"):
        y, s = [], []
        for g in (pg, *neg):
            lab = 1 if g == pg else 0
            for f in F[g]:
                y.append(lab)
                s.append(f[name])
        print(f"{name:12s} AUC {pg:17s} vs all OOS: {roc_auc_score(y, s):.3f}")
for g, fs in F.items():
    for name in ("dense_top1", "dense_mean5"):
        v = sorted(f[name] for f in fs)
        print(
            f"{g:18s} {name:11s} min {v[0]:.3f} p25 {v[len(v) // 4]:.3f} median {v[len(v) // 2]:.3f} max {v[-1]:.3f}"
        )
