"""Does down-weighting corpus-unknown *common English* words fix verbose over-abstention without hurting out-of-scope abstention?

Development data only (never the frozen held-out or final-evaluation sets). Threshold chosen with the Milestone 5 rule:
largest abstain threshold whose false-abstain rate on *train authored* answerable queries is <= 5%, maximising abstention on
the calibration half of the unanswerable set.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from neural_debugger.rag.coverage_eval import build_pipeline
from neural_debugger.rag.evaluation import load_items
from neural_debugger.rag.evidence import select_evidence
from neural_debugger.rag.sufficiency import SufficiencyConfig, TermStats, assess

VOCAB = next(
    Path.home().glob(".cache/huggingface/hub/models--thenlper--gte-small/snapshots/*/vocab.txt")
)
COMMON = {
    w.strip()
    for w in VOCAB.read_text(encoding="utf-8").splitlines()
    if re.fullmatch(r"[a-z]{3,}", w.strip())
}
COMMON |= {"am", "ve", "ll", "don", "isn", "doesn", "didn", "won"}


class FillerAwareStats(TermStats):
    def __init__(self, base: TermStats, unknown_common_weight: float) -> None:
        super().__init__(base.df, base.n)
        self.w = unknown_common_weight

    def idf(self, term: str) -> float:
        if self.df.get(term, 0) == 0 and term in COMMON:
            return self.w
        return super().idf(term)


pipe = build_pipeline(
    ["data/corpus", "data/corpus_supplement"],
    "data/embeddings_rag",
    "hybrid-gte-tuned",
    "cpu",
    "coverage",
)
base_stats = pipe.stats
items = load_items()
vd = yaml.safe_load(open("data/eval/rag/verbose_dev.yaml"))["questions"]
cd = yaml.safe_load(open("data/eval/rag/coverage_dev.yaml"))["questions"]
tb = yaml.safe_load(open("data/eval/rag/traceback_dev.yaml"))["questions"]

# one (question info, gate evidence) per text, computed once
texts: dict[str, tuple] = {}


def prep(text: str):
    if text not in texts:
        p = pipe.prepare(text, None)
        ev, _ = select_evidence(p.sections[: pipe.config.gate_k], pipe.config.gate_evidence)
        texts[text] = (p.question, ev)
    return texts[text]


def cov(text: str, stats: TermStats) -> float:
    q, ev = prep(text)
    return assess(q, ev, stats, SufficiencyConfig()).coverage


groups = {
    "train_answerable(n=%d)": [
        i.query for i in items if i.group == "authored" and i.split == "train"
    ],
    "val_answerable": [i.query for i in items if i.group == "authored" and i.split == "val"],
    "unans_calib": [i.query for i in items if i.group == "unanswerable" and i.split == "calib"],
    "unans_heldout_half": [
        i.query for i in items if i.group == "unanswerable" and i.split == "heldout"
    ],
    "dev_oos(own+ind)": [q["q"] for q in cd if q["kind"] == "abstain"],
    "verbose_oos": [q["q"] for q in vd if q["kind"] == "abstain"],
    "verbose_in_scope": [q["q"] for q in vd if q["kind"] == "answer"],
    "traceback_dev": [q["q"] for q in tb],
    "dev_in_scope(own+ind, verbose)": [q["q"] for q in cd if q["kind"] == "answer"],
}
from neural_debugger.rag.sufficiency import salient_terms
from neural_debugger.rag.text import term_set


def cov_topk(text: str, st: TermStats, k: int | None, power: float, filler: bool) -> float:
    q, ev = prep(text)
    wanted = salient_terms(q, 40)
    have: set[str] = set()
    for e in ev:
        have |= term_set(" ".join([e.title, e.heading, e.text]))
    w = {}
    for t in wanted:
        if filler and st.df.get(t, 0) == 0 and t in COMMON:
            continue
        w[t] = st.idf(t) ** power
    terms_ = sorted(w, key=lambda t: -w[t])[:k] if k else list(w)
    tot = sum(w[t] for t in terms_)
    return sum(w[t] for t in terms_ if t in have) / tot if tot else 0.0


import sys

if len(sys.argv) > 1 and sys.argv[1] == "slip":
    thr_old, thr_new = 0.375, 0.275
    for g in ("unans_calib", "unans_heldout_half", "dev_oos(own+ind)", "verbose_oos"):
        for t in groups[g]:
            old = cov_topk(t, base_stats, None, 1.0, False)
            new = cov_topk(t, base_stats, None, 2.0, True)
            if old < thr_old and new >= thr_new:
                q, ev = prep(t)
                unknown = [x for x in salient_terms(q, 40) if base_stats.df.get(x, 0) == 0]
                print(
                    f"[{g}] old={old:.2f} new={new:.2f} :: {t[:90]!r}\n      unknown terms: {[(x, 'COMMON' if x in COMMON else 'tech') for x in unknown][:10]}"
                )
    raise SystemExit
configs = [("M5 (all terms, p=1)", None, 1.0, False), ("filler-free, all terms", None, 1.0, True)]
for k in (8, 10, 12, 16):
    configs.append((f"filler-free, top-{k} IDF", k, 1.0, True))
configs.append(("top-12 IDF, filler kept", 12, 1.0, False))
configs.append(("filler-free, all terms, idf^2", None, 2.0, True))
for name, k, power, filler in configs:
    c = {g: [cov_topk(t, base_stats, k, power, filler) for t in ts] for g, ts in groups.items()}
    tr = sorted(c["train_answerable(n=%d)"])
    best = None
    for kk in range(0, 41):
        thr = round(kk * 0.025, 3)
        fa = sum(x < thr for x in tr) / len(tr)
        ab = sum(x < thr for x in c["unans_calib"]) / len(c["unans_calib"])
        if fa <= 0.05 and (best is None or (ab, thr) >= (best[1], best[0])):
            best = (thr, ab, fa)
    thr = best[0]
    row = {g: sum(x < thr for x in xs) for g, xs in c.items()}
    print(
        f"{name:32s} thr={thr:<5} FA(train)={best[2]:.3f} | unans {row['unans_calib']}/16+{row['unans_heldout_half']}/16 | dev_oos {row['dev_oos(own+ind)']}/9 | verbose_oos {row['verbose_oos']}/12 | VERBOSE in-scope abstain {row['verbose_in_scope']}/36 | tb {row['traceback_dev']}/10 | dev in-scope {row['dev_in_scope(own+ind, verbose)']}/39"
    )
