"""Irrelevant-but-valid citations: does the support check accept a claim against a *nearby different* section?

Pairs: data/eval/rag/citation_relevance_dev.jsonl (Gemini-paraphrased claim; true source section; dense-nearest other section that
lacks most of the claim's distinctive terms). A judge should call (claim, true) supported and should NOT call (claim, adjacent)
supported. Dev half chooses settings; the test half is reported. Model-generated pairs, not human-verified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from neural_debugger.data.corpus import load_corpus_overlay
from neural_debugger.rag.sufficiency import TermStats
from neural_debugger.rag.support import LexicalSupportJudge
from neural_debugger.rag.text import term_set

corpus = load_corpus_overlay(["data/corpus", "data/corpus_supplement"])
stats = TermStats.from_texts(c.index_text() for c in corpus.chunks)
rows = [json.loads(line) for line in open("data/eval/rag/citation_relevance_dev.jsonl")]
half = lambda i: "dev" if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 == 0 else "test"


def passage(sid: str) -> str:
    # what validate.py gives the judge: "title > heading\ntext" with the evidence text capped at 1500 characters
    title = corpus.section_heading_path(sid)
    t = corpus.section_text(sid)[:1500]
    return f"{title[0]} > {' > '.join(title)}\n{t}"


def idf_cov(claim: str, text: str, floor: float = 0.0) -> float:
    ct = [t for t in term_set(claim) if stats.idf(t) >= floor]
    have = term_set(text)
    tot = sum(stats.idf(t) for t in ct) or 1.0
    return sum(stats.idf(t) for t in ct if t in have) / tot


def evaluate(name, fn):
    out = {}
    for h in ("dev", "test"):
        sub = [r for r in rows if half(r["id"]) == h]
        true_ok = sum(fn(r["claim"], passage(r["true_section"])) == "supported" for r in sub)
        adj_sup = sum(fn(r["claim"], passage(r["adjacent_section"])) == "supported" for r in sub)
        adj_not_rej = sum(
            fn(r["claim"], passage(r["adjacent_section"])) != "unsupported" for r in sub
        )
        true_rej = sum(fn(r["claim"], passage(r["true_section"])) == "unsupported" for r in sub)
        out[h] = (len(sub), true_ok, true_rej, adj_sup, adj_not_rej)
    d, t = out["dev"], out["test"]
    print(
        f"{name:34s} dev n={d[0]}: true supported {d[1] / d[0]:.2f} rejected {d[2] / d[0]:.2f} | adjacent SUPPORTED {d[3] / d[0]:.2f} not-rejected {d[4] / d[0]:.2f}  ||  test n={t[0]}: true supported {t[1] / t[0]:.2f} rejected {t[2] / t[0]:.2f} | adjacent SUPPORTED {t[3] / t[0]:.2f} not-rejected {t[4] / t[0]:.2f}"
    )


lex = LexicalSupportJudge()
evaluate("current lexical judge", lambda c, p: lex.judge(c, [p]).label)
if len(sys.argv) > 1:
    for tau in (0.5, 0.6, 0.7, 0.8):
        evaluate(
            f"idf-weighted coverage >= {tau}",
            lambda c, p, tau=tau: (
                "supported"
                if idf_cov(c, p) >= tau
                else ("partial" if idf_cov(c, p) >= tau - 0.2 else "unsupported")
            ),
        )
    for fl in (3.0, 4.0, 5.0):
        for tau in (0.6, 0.75):
            evaluate(
                f"idf>= {fl} terms only, cov >= {tau}",
                lambda c, p, tau=tau, fl=fl: (
                    "supported"
                    if idf_cov(c, p, fl) >= tau
                    else ("partial" if idf_cov(c, p, fl) >= tau - 0.25 else "unsupported")
                ),
            )


if len(sys.argv) > 1 and sys.argv[1] == "dense":
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config

    fac = SystemFactory(
        corpus,
        load_neural_config("configs/neural.yaml", "configs/m4.yaml"),
        device="cpu",
        embedding_dir=Path("data/embeddings_rag"),
    )
    dense = fac.build("dense-gte-small")
    by_sec: dict[str, list[int]] = {}
    for i, c in enumerate(dense.chunks):
        by_sec.setdefault(c.section_id, []).append(i)
    vecs = np.load(
        "data/embeddings_rag/gte-small/embeddings.npy"
    )  # one row per overlay chunk, in corpus order
    assert len(vecs) == len(dense.chunks)

    def dsim(claim: str, sid: str, qv=None) -> float:
        qv = dense.encode_query(claim) if qv is None else qv
        return float(max(vecs[j] @ qv for j in by_sec[sid]))

    feats = []
    for r in rows:
        qv = dense.encode_query(r["claim"])
        feats.append(
            {
                "id": r["id"],
                "t_dense": dsim(r["claim"], r["true_section"], qv),
                "a_dense": dsim(r["claim"], r["adjacent_section"], qv),
                "t_lex": idf_cov(r["claim"], passage(r["true_section"])),
                "a_lex": idf_cov(r["claim"], passage(r["adjacent_section"])),
                "t_old": {"supported": 2, "partial": 1, "unsupported": 0}[
                    lex.judge(r["claim"], [passage(r["true_section"])]).label
                ],
                "a_old": {"supported": 2, "partial": 1, "unsupported": 0}[
                    lex.judge(r["claim"], [passage(r["adjacent_section"])]).label
                ],
            }
        )
    for name, tk, ak in (
        ("dense max-chunk cosine", "t_dense", "a_dense"),
        ("idf-weighted lexical coverage", "t_lex", "a_lex"),
        ("current judge (0/1/2)", "t_old", "a_old"),
    ):
        y = [1] * len(feats) + [0] * len(feats)
        s = [f[tk] for f in feats] + [f[ak] for f in feats]
        print(f"AUC true-vs-adjacent  {name:30s} {roc_auc_score(y, s):.3f}")
    for name in ("t_dense", "a_dense"):
        v = sorted(f[name] for f in feats)
        print(
            f"{name}: min {v[0]:.3f} p10 {v[len(v) // 10]:.3f} p25 {v[len(v) // 4]:.3f} median {v[len(v) // 2]:.3f} p75 {v[3 * len(v) // 4]:.3f} max {v[-1]:.3f}"
        )
    json.dump(feats, open("/tmp/nd_cr_feats.json", "w"))
