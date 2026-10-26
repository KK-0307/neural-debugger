"""Build and print the unjudged candidate pool contributed by neural/hybrid systems (dev queries).

    uv run python scripts/pool_neural_candidates.py build   --depth 5 --rankings 'data/eval/results/m3/rankings/dev_*.jsonl' --out data/eval/results/m3/pool_dev_round1.json
    uv run python scripts/pool_neural_candidates.py show    --pool data/eval/results/m3/pool_dev_round1.json --start 0 --count 35
    uv run python scripts/pool_neural_candidates.py record  --pool ... --judgments judgments.txt   # appends to data/eval/pooled_judgments_neural.yaml

Judgments file lines: ``qid: 2 5`` = candidates 2 and 5 are relevant; every shown candidate counts
as reviewed (so the rest were judged not relevant). Official-FAQ queries are never pooled.
"""

import argparse
import glob
import json
from pathlib import Path

import yaml

from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.queries import load_materialised
from neural_debugger.ml.evaluation.qrels import load_qrels

POOLED = Path("data/eval/pooled_judgments_neural.yaml")


def build(
    depth: int, pattern: str, out: Path, splits: tuple[str, ...], variant: str = "expanded"
) -> None:
    corpus = load_corpus(Path("data/corpus"))
    qs = [
        q
        for q in load_materialised(Path("data/eval/queries.jsonl"))
        if q.split in splits and q.slice != "official_faq"
    ]
    qr = load_qrels(qs, corpus, variant)  # already-judged = original pool + earlier neural rounds
    cand: dict[str, list[str]] = {q.id: [] for q in qs}
    for f in sorted(glob.glob(pattern)):
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            if r["id"] not in cand:
                continue
            for s in r["ranking"][:depth]:
                if s not in qr.judged[r["id"]] and s not in cand[r["id"]]:
                    cand[r["id"]].append(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cand, indent=0, sort_keys=True))
    print(sum(len(v) for v in cand.values()), "candidates for", len(cand), "queries ->", out)


def show(pool: Path, start: int, count: int) -> None:
    corpus = load_corpus(Path("data/corpus"))
    qmap = {q.id: q for q in load_materialised(Path("data/eval/queries.jsonl"))}
    cand = json.loads(pool.read_text())
    for qid in sorted(cand)[start : start + count]:
        print(f"\n[{qid}] {qmap[qid].query[:120]}")
        for i, s in enumerate(cand[qid], start=1):
            body = " ".join(corpus.section_text(s).split())[:150]
            print(f"  {i} {s.split('/', 2)[2][:70]} :: {body}")


def record(pool: Path, judgments: Path, target: Path = POOLED) -> None:
    cand = json.loads(pool.read_text())
    existing = yaml.safe_load(target.read_text()) if target.exists() else {}
    existing = existing or {}
    added_total = 0
    shown = {q for q in cand}
    seen_in_file = set()
    for line in judgments.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        qid, _, rest = line.partition(":")
        qid = qid.strip()
        seen_in_file.add(qid)
        idx = [int(x) for x in rest.split()]
        assert all(1 <= i <= len(cand[qid]) for i in idx), (qid, idx)
        entry = existing.setdefault(qid, {"added": [], "reviewed": []})
        entry["added"] = sorted(set(entry["added"]) | {cand[qid][i - 1] for i in idx})
        added_total += len(idx)
    # every candidate shown in this pool counts as reviewed, judged relevant or not
    for qid in shown:
        entry = existing.setdefault(qid, {"added": [], "reviewed": []})
        entry["reviewed"] = sorted(set(entry["reviewed"]) | set(cand[qid]))
    header = (
        "# Relevance judgments from expanding the pool with neural/hybrid candidates (Milestone 3).\n"
        "# `reviewed` = every candidate read; `added` = judged relevant. Judged by the assistant; not human-verified.\n"
    )
    target.write_text(header + yaml.safe_dump(existing, sort_keys=True, width=240))
    print(
        f"recorded {added_total} relevant additions over {len(seen_in_file)} queries; {len(shown)} queries marked reviewed"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "show", "record"])
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--rankings", default="data/eval/results/m3/rankings/dev_*.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--pool", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=35)
    ap.add_argument("--judgments", type=Path)
    ap.add_argument("--target", type=Path, default=POOLED)
    ap.add_argument("--variant", default="expanded", choices=["expanded", "m4"])
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.depth, a.rankings, a.out, tuple(a.splits), a.variant)
    elif a.cmd == "show":
        show(a.pool, a.start, a.count)
    else:
        record(a.pool, a.judgments, a.target)
