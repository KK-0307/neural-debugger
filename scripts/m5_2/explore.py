import re
import sys
from pathlib import Path

import yaml

from neural_debugger.data.queries import load_materialised
from neural_debugger.ml.evaluation.metrics import hit_at_k, ndcg_at_k
from neural_debugger.rag.coverage_eval import build_pipeline
from neural_debugger.rag.question import prepare_question
from neural_debugger.rag.text import term_set, terms

pipe = build_pipeline(["data/corpus"], None, "hybrid-gte-tuned", "cpu")
stats = pipe.stats
ret = pipe.retriever
gold_items = [
    i
    for i in yaml.safe_load(open("data/eval/rag/coverage_dev.yaml"))["questions"]
    if i["gold"] and not any(g.get("supplement") for g in i["gold"])
]
dev = [q for q in load_materialised(Path("data/eval/queries.jsonl")) if q.split in ("train", "val")]
print(len(gold_items), "gold items;", len(dev), "dev queries")

FILLER = set(
    """massive efficiently correctly properly best way recommended causing despite keeps seems anything something someone
just really actually basically very quite maybe possibly currently already still even though however also tried trying try
want need needs help please issue problem question wondering getting get got use using used work works working
properly appropriate effective efficient simple easy common typical typically usually often whenever without within
whole entire single custom specific different certain particular""".split()
)
vocab = {t for t, d in stats.df.items() if d >= 2}


def split_compound(t):
    if len(t) < 7 or t in vocab:
        return []
    for i in range(3, len(t) - 2):
        a, b = t[:i], t[i:]
        if a in vocab and b in vocab:
            return [a, b]
    return []


def normalize(text):
    extra = []
    for t in set(re.findall(r"[A-Za-z]{7,}", text)):
        sp = split_compound(t.lower())
        if sp:
            extra.append(" ".join(sp))
    return text + (" " + " ".join(extra) if extra else "")


def sal(text):
    out = []
    for t in terms(text):
        if t in FILLER or t.isdigit():
            continue
        if t not in out:
            out.append(t)
    return out


cache = {}


def section_terms(s):
    k = s.section_id
    if k not in cache:
        cache[k] = term_set(" ".join([s.title, s.heading, s.text[:4000]]))
    return cache[k]


def lex_scores(q, cands):
    W = sal(q)
    tot = sum(stats.idf(t) for t in W) or 1
    return {
        s.section_id: sum(stats.idf(t) for t in W if t in section_terms(s)) / tot for s in cands
    }


def rank(q, mode, K=40, w=1.0, rrfk=20):
    qq = normalize(q) if "norm" in mode else q
    cands = ret.search(qq, K, None)
    if "lex" not in mode:
        return [s.section_id for s in cands]
    ls = lex_scores(q, cands)
    order = sorted(cands, key=lambda s: -ls[s.section_id])
    lr = {s.section_id: i + 1 for i, s in enumerate(order)}
    fused = {s.section_id: 1 / (rrfk + s.rank) + w / (rrfk + lr[s.section_id]) for s in cands}
    return sorted(fused, key=lambda k: -fused[k])


modes = ["base", "norm", "lex", "norm+lex"]
if len(sys.argv) > 1:
    modes = sys.argv[1:]
for mode in modes:
    for w in [1.0] if "lex" in mode else [0]:
        ranks = []
        for it in gold_items:
            q = prepare_question(it["q"]).retrieval_query
            r = rank(q, mode, w=w)
            best = min(
                [r.index(g["section"]) + 1 for g in it["gold"] if g["section"] in r] or [999]
            )
            ranks.append(best)
        nd = []
        h5 = []
        for rec in dev:
            r = rank(rec.query, mode, K=100 if "lex" not in mode else 40, w=w)
            rel = set(rec.relevant)
            nd.append(ndcg_at_k(r, rel, 5))
            h5.append(hit_at_k(r, rel, 5))
        print(
            f"{mode:10s} w={w}: gold ranks {ranks} | within5={sum(x <= 5 for x in ranks)} within8={sum(x <= 8 for x in ranks)} | dev NDCG@5={sum(nd) / len(nd):.3f} hit@5={sum(h5) / len(h5):.3f}"
        )

# ---- multi-query decomposition
_SENT = re.compile(r"(?<=[.!?])\s+")


def subqueries(raw, include_full=True, max_sub=4):
    info = prepare_question(raw)
    out = []
    if include_full:
        out.append(info.retrieval_query)
    for e in info.error_lines[-1:]:
        if e not in out:
            out.append(e)
    for s in _SENT.split(info.retrieval_query):
        s = s.strip()
        if len(s.split()) >= 4 and s not in out and not any(s in o for o in out[1:]):
            out.append(s)
    return out[: max_sub + 1]


def rank_mq(raw, K=30, rrfk=60, include_full=True, wfull=1.0):
    subs = subqueries(raw, include_full)
    fused = {}
    for i, sq in enumerate(subs):
        w = wfull if (i == 0 and include_full) else 1.0
        for s in ret.search(sq, K, None):
            fused[s.section_id] = fused.get(s.section_id, 0) + w / (rrfk + s.rank)
    return sorted(fused, key=lambda k: -fused[k])


for name, kw in [
    ("mq(full+subs)", dict()),
    ("mq(subs only)", dict(include_full=False)),
    ("mq(full x2)", dict(wfull=2.0)),
]:
    ranks = []
    for it in gold_items:
        r = rank_mq(it["q"], **kw)
        ranks.append(
            min([r.index(g["section"]) + 1 for g in it["gold"] if g["section"] in r] or [999])
        )
    nd = []
    h5 = []
    for rec in dev:
        r = rank_mq(rec.query, **kw)
        rel = set(rec.relevant)
        nd.append(ndcg_at_k(r, rel, 5))
        h5.append(hit_at_k(r, rel, 5))
    print(
        f"{name:14s}: gold ranks {ranks} | within5={sum(x <= 5 for x in ranks)} within8={sum(x <= 8 for x in ranks)} | dev NDCG@5={sum(nd) / len(nd):.3f} hit@5={sum(h5) / len(h5):.3f}"
    )
