"""Pre-LLM diagnosis of the evidence gate on the Milestone 5.3 development set (verbose, section-seeded; no model calls).

python scripts/m5_3/diagnose_gate.py --out data/eval/results/m5_3/gate_diagnosis_before.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from neural_debugger.rag.coverage_eval import build_pipeline
from neural_debugger.rag.text import term_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/eval/rag/verbose_dev.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pipe = build_pipeline(
        ["data/corpus", "data/corpus_supplement"],
        "data/embeddings_rag",
        "hybrid-gte-tuned",
        "cpu",
        "coverage",
    )
    rows = []
    for q in yaml.safe_load(open(args.dev))["questions"]:
        prep = pipe.prepare(q["q"], None)
        gold = q.get("gold_section")
        retrieved = {s.section_id: s.rank for s in prep.sections}
        shown = [e.section_id for e in prep.evidence]
        suff = prep.sufficiency
        row = {
            "id": q["id"],
            "kind": q["kind"],
            "fw": q["fw"],
            "style": q["style"],
            "gate": suff.level,
            "coverage": round(suff.coverage, 3),
            "missing_terms": suff.missing_terms,
            "gold_rank": retrieved.get(gold) if gold else None,
            "gold_in_evidence": gold in shown if gold else None,
            "n_query_terms": len(term_set(prep.question.gate_text)),
        }
        rows.append(row)
    ans = [r for r in rows if r["kind"] == "answer"]
    oos = [r for r in rows if r["kind"] == "abstain"]
    summary = {
        "in_scope": len(ans),
        "gate_abstain_in_scope": sum(r["gate"] == "none" for r in ans),
        "gold_retrieved": sum(r["gold_rank"] is not None for r in ans),
        "gold_in_evidence": sum(bool(r["gold_in_evidence"]) for r in ans),
        "gate_abstain_though_gold_in_evidence": sum(
            r["gate"] == "none" and r["gold_in_evidence"] for r in ans
        ),
        "oos": len(oos),
        "oos_gate_abstain": sum(r["gate"] == "none" for r in oos),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["kind"] == "answer" and r["gate"] == "none":
            print(
                f"{r['id']:5s} {r['style']:9s} cov={r['coverage']:.2f} gold_ev={r['gold_in_evidence']} missing={r['missing_terms']}"
            )


if __name__ == "__main__":
    main()
