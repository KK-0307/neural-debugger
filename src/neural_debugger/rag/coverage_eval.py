"""Where do in-scope questions get lost before the model is ever called? (Milestone 5.2 diagnosis tool)

For every annotated development question it runs the pipeline up to the evidence gate (``RAGPipeline.prepare``, the same
code the live system uses, no model call) and says which stage lost the answer:

* ``no_answer_in_corpus``  the annotators found no answer-bearing text in the indexed documentation;
* ``corpus_gap``           the answer-bearing section is not in the corpus that is loaded (e.g. the base corpus only);
* ``not_retrieved``        the section is in the corpus but not among the retrieved candidates (ranking failure);
* ``evidence_dropped``     retrieved, but evidence selection did not show it to the model;
* ``text_truncated``       shown, but the answer-bearing text was cut off by the per-section budget;
* ``gate_blocked``         the evidence is right but the sufficiency gate would abstain before calling the model;
* ``reachable``            the model would be shown the answer-bearing text.

Annotations are agent-made (searching the corpus), not human, and these are development questions: they guide changes and
are never the held-out evaluation. Out-of-scope questions are reported separately (the gate must keep abstaining on them).

    python -m neural_debugger.rag.coverage_eval --out data/eval/results/m5_2/coverage_dev_before.json
    python -m neural_debugger.rag.coverage_eval --corpus-dir data/corpus data/corpus_supplement \\
        --embedding-dir data/embeddings_rag --out data/eval/results/m5_2/coverage_dev_after.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .pipeline import PipelineConfig, RAGPipeline
from .providers.testing import ExtractiveProvider
from .sufficiency import TermStats

DEV = Path("data/eval/rag/coverage_dev.yaml")
TRACEBACKS = Path("data/eval/rag/traceback_dev.yaml")


def build_pipeline(
    corpus_dirs: list[str],
    embedding_dir: str | None,
    retriever: str,
    device: str,
    profile: str = "m5",
) -> RAGPipeline:
    from neural_debugger.data.corpus import load_corpus_overlay
    from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config

    from .retrieval import CorpusSectionRetriever

    corpus = load_corpus_overlay(corpus_dirs)
    cfg = load_neural_config("configs/neural.yaml", "configs/m4.yaml")
    system = SystemFactory(
        corpus, cfg, device=device, embedding_dir=Path(embedding_dir) if embedding_dir else None
    ).build(retriever)
    stats = TermStats.from_texts(c.index_text() for c in corpus.chunks)
    return RAGPipeline(
        CorpusSectionRetriever(system, corpus),
        ExtractiveProvider(),
        PipelineConfig.coverage_profile() if profile == "coverage" else PipelineConfig(),
        stats=stats,
    )


def diagnose_item(pipe: RAGPipeline, item: dict[str, Any]) -> dict[str, Any]:
    prep = pipe.prepare(item["q"], None)
    corpus_sections = pipe.retriever._known  # type: ignore[attr-defined]  # noqa: SLF001
    retrieved = {s.section_id: s.rank for s in prep.sections}
    shown = {e.section_id: e for e in prep.evidence}
    row: dict[str, Any] = {
        "id": item["id"],
        "kind": item["kind"],
        "gate": prep.sufficiency.level,
        "coverage": round(prep.sufficiency.coverage, 3),
        "missing_terms": prep.sufficiency.missing_terms[:6],
        "retrieval_query": prep.question.retrieval_query,
        "evidence": [e.section_id for e in prep.evidence],
        "retrieval_ms": round(prep.retrieval_ms, 1),
    }
    if item["kind"] == "abstain":
        row["stage"] = "out_of_scope"
        return row
    if item.get("no_answer_in_corpus"):
        row["stage"] = "no_answer_in_corpus"
        return row
    gold = item.get("gold") or []
    if not gold:
        row["stage"] = "unannotated"
        return row
    best = "corpus_gap"
    order = [
        "corpus_gap",
        "not_retrieved",
        "evidence_dropped",
        "text_truncated",
        "gate_blocked",
        "reachable",
    ]
    detail = []
    for g in gold:
        sid = g["section"]
        if sid not in corpus_sections:
            stage = "corpus_gap"
        elif sid not in retrieved:
            stage = "not_retrieved"
        elif sid not in shown:
            stage = "evidence_dropped"
        elif not re.search(g["key"], shown[sid].text, re.I | re.S):
            stage = "text_truncated"
        elif prep.sufficiency.level == "none":
            stage = "gate_blocked"
        else:
            stage = "reachable"
        detail.append(
            {
                "section": sid,
                "stage": stage,
                "rank": retrieved.get(sid),
                "partial": g.get("partial", False),
            }
        )
        if order.index(stage) > order.index(best):
            best = stage
    row["stage"] = best
    row["gold"] = detail
    return row


def traceback_eval(pipe: RAGPipeline, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Doc-level gold: did the pages that explain the error make it into the candidates / the evidence?"""
    rows = []
    for it in items:
        prep = pipe.prepare(it["q"], None)
        gold = set(it["gold_docs"])
        ranks = [s.rank for s in prep.sections if s.section_id.split("#", 1)[0] in gold]
        shown = [e for e in prep.evidence if e.section_id.split("#", 1)[0] in gold]
        rows.append(
            {
                "id": it["id"],
                "first_gold_rank": min(ranks) if ranks else None,
                "gold_in_evidence": bool(shown),
                "gate": prep.sufficiency.level,
                "query": prep.question.retrieval_query,
            }
        )
    n = len(rows)
    return {
        "n": n,
        "gold_in_top5": sum(bool(r["first_gold_rank"]) and r["first_gold_rank"] <= 5 for r in rows),
        "gold_in_candidates": sum(r["first_gold_rank"] is not None for r in rows),
        "gold_in_evidence": sum(r["gold_in_evidence"] for r in rows),
        "gate_abstain": sum(r["gate"] == "none" for r in rows),
        "rows": rows,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["kind"] == "answer"]
    oos = [r for r in rows if r["kind"] == "abstain"]
    stages = Counter(r["stage"] for r in ans)
    gold_rows = [r for r in ans if r["stage"] not in ("no_answer_in_corpus", "unannotated")]
    return {
        "n_in_scope": len(ans),
        "stages_in_scope": dict(stages),
        "gold_items": len(gold_rows),
        "gold_reachable": sum(r["stage"] == "reachable" for r in gold_rows),
        "in_scope_gate_abstain": sum(r["gate"] == "none" for r in ans),
        "in_scope_gate_pass_rate": round(sum(r["gate"] != "none" for r in ans) / len(ans), 3)
        if ans
        else None,
        "n_out_of_scope": len(oos),
        "out_of_scope_gate_abstain": sum(r["gate"] == "none" for r in oos),
        "retrieval_ms_p50": sorted(r["retrieval_ms"] for r in rows)[len(rows) // 2]
        if rows
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dev", type=Path, default=DEV)
    ap.add_argument("--corpus-dir", nargs="+", default=["data/corpus"])
    ap.add_argument("--embedding-dir", default=None)
    ap.add_argument("--retriever", default="hybrid-gte-tuned")
    ap.add_argument("--profile", choices=["m5", "coverage"], default="m5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tracebacks", type=Path, default=TRACEBACKS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    items = yaml.safe_load(args.dev.read_text())["questions"]
    started = time.perf_counter()
    pipe = build_pipeline(
        args.corpus_dir, args.embedding_dir, args.retriever, args.device, args.profile
    )
    rows = [diagnose_item(pipe, it) for it in items]
    report = {
        "label": "development diagnosis; agent-annotated gold; not held-out, not human",
        "corpus_dirs": args.corpus_dir,
        "retriever": args.retriever,
        "profile": args.profile,
        "summary": summarize(rows),
        "tracebacks": traceback_eval(
            pipe, yaml.safe_load(args.tracebacks.read_text())["questions"]
        ),
        "rows": rows,
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))
    tb = report["tracebacks"]
    print({k: v for k, v in tb.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
