"""Re-judge recorded live-model answers offline (no API calls).

``live_acceptance`` stores the model's raw replies. This module replays each reply through the same retrieval
and evidence selection, then asks every support judge about every claim the model made (including the ones
the pipeline dropped), so the lexical check, the semantic rules and the NLI stage can be compared on *real*
model paraphrases and reviewed by hand. The output is an inspection table, not a score: nobody has labelled
these claims.

    python -m neural_debugger.rag.live_replay data/eval/results/m5_1/live_gemini_gemini-3.1-flash-lite.raw.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .live_acceptance import (
    POISON_PARAPHRASE,
    POISON_REGEX,
    QUESTIONS,
    ItemResult,
    PoisonedRetriever,
    judge_item,
    summarize,
)
from .parse import parse_model_output
from .pipeline import PipelineConfig, RAGPipeline
from .providers.testing import ScriptedProvider
from .retrieval import SectionRetriever
from .semantic import CrossEncoderNLI, GroundedSupportJudge, SemanticConfig, SemanticGuard
from .sufficiency import TermStats
from .support import LexicalSupportJudge, SupportJudge
from .types import Evidence

_POISON = {"injection_doc": POISON_REGEX, "injection_doc_paraphrase": POISON_PARAPHRASE}


def _claims(reply: str) -> list[tuple[str, str, list[str]]]:
    parsed = parse_model_output(reply).answer
    if parsed is None or parsed.status == "insufficient_evidence":
        return []
    out: list[tuple[str, str, list[str]]] = []
    if parsed.diagnosis:
        out.append(("diagnosis", parsed.diagnosis.text, list(parsed.diagnosis.citations)))
    out += [("step", s.text, list(s.citations)) for s in parsed.steps]
    out += [
        ("code", c.explanation or "Suggested code.", list(c.citations))
        for c in parsed.code_suggestions
    ]
    out += [("conflict", c.text, list(c.citations)) for c in parsed.conflicts]
    return out


def replay(
    raw: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    retriever: SectionRetriever,
    stats: TermStats,
    judges: dict[str, SupportJudge],
) -> list[dict[str, Any]]:
    by_id = {q["id"]: q for q in questions}
    rows: list[dict[str, Any]] = []
    for item in raw:
        calls = [c for c in item["calls"] if "reply" in c and c.get("purpose") == "answer"]
        if not calls:
            continue
        q = by_id[item["id"]]
        retr = retriever
        if q["kind"] in _POISON:
            retr = PoisonedRetriever(retriever, _POISON[q["kind"]])
        reply = calls[0]["reply"]
        pipe = RAGPipeline(
            retr, ScriptedProvider([reply]), PipelineConfig(semantic="off"), stats=stats
        )
        evidence: dict[str, Evidence] = {e.id: e for e in pipe.answer(q["q"], None).evidence}
        for kind, text, cites in _claims(reply):
            good = [c for c in cites if c in evidence]
            passages = [
                f"{evidence[c].title} > {evidence[c].heading}\n{evidence[c].text}" for c in good
            ]
            row: dict[str, Any] = {
                "id": item["id"],
                "claim_kind": kind,
                "claim": text,
                "citations": good,
                "verdicts": {},
            }
            for name, judge in judges.items():
                v = (
                    judge.judge(text, passages, trusted=q["q"])  # type: ignore[call-arg]
                    if passages and getattr(judge, "supports_trusted", False)
                    else (judge.judge(text, passages) if passages else None)
                )
                row["verdicts"][name] = (
                    None
                    if v is None
                    else {"label": v.label, "semantic": v.semantic, "reason": v.reason[:160]}
                )
            rows.append(row)
    return rows


def merge_raw(raws: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One raw record per question id: the first run that got a model response; reruns only fill errors."""
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raws:
        for item in raw:
            old = by_id.get(item["id"])
            if old is None or (
                old["result"]["status"] == "error" and item["result"]["status"] != "error"
            ):
                by_id[item["id"]] = item
    return list(by_id.values())


def rescore(
    raw: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    retriever: SectionRetriever,
    stats: TermStats,
    config: PipelineConfig,
) -> list[ItemResult]:
    """Replay each recorded model reply through the *current* pipeline and re-derive the validation outcomes.

    What the model did (replies, tokens, latency) is kept from the live run; what the validators decide is
    recomputed with the code as it is now. Items with no model response keep their recorded result.
    """
    by_id = {q["id"]: q for q in questions}
    out: list[ItemResult] = []
    for item in raw:
        orig = ItemResult(**item["result"])
        replies = [c["reply"] for c in item["calls"] if "reply" in c]
        if not replies:
            out.append(orig)
            continue
        q = by_id[item["id"]]
        retr = (
            PoisonedRetriever(retriever, _POISON[q["kind"]]) if q["kind"] in _POISON else retriever
        )
        ans = RAGPipeline(retr, ScriptedProvider(replies), config, stats=stats).answer(q["q"], None)
        new = judge_item(q, ans, replies)
        new.generation_ms, new.total_ms, new.tokens = orig.generation_ms, orig.total_ms, orig.tokens
        out.append(new)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "raw", type=Path, nargs="+", help="raw.json files from live_acceptance (later = reruns)"
    )
    ap.add_argument(
        "--rescore",
        choices=["off", "rules", "nli"],
        help="re-derive validation outcomes with the current code and this semantic mode, then print the summary",
    )
    ap.add_argument("--rescore-out", type=Path)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--retriever", default="hybrid-gte-tuned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--config-from",
        type=Path,
        default=Path("data/eval/results/m5_1/semantic_grounding.json"),
        help="grounding_eval report whose dev-calibrated ``chosen_config`` is also replayed",
    )
    args = ap.parse_args(argv)

    from neural_debugger.data.corpus import load_corpus
    from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config

    from .retrieval import CorpusSectionRetriever

    raws = merge_raw([json.loads(p.read_text()) for p in args.raw])
    corpus = load_corpus("data/corpus")
    system = SystemFactory(
        corpus, load_neural_config("configs/neural.yaml", "configs/m4.yaml"), device=args.device
    ).build(args.retriever)
    stats = TermStats.from_texts(c.index_text() for c in corpus.chunks)
    questions = yaml.safe_load(args.questions.read_text())["questions"]
    section_retriever = CorpusSectionRetriever(system, corpus)
    if args.rescore:
        results = rescore(
            raws, questions, section_retriever, stats, PipelineConfig(semantic=args.rescore)
        )
        summary = summarize(
            results,
            sum(1 for r in raws for c in r["calls"] if "reply" in c or "error" in c),
            0.0,
        )
        summary["label"] = (
            f"REPLAY of recorded model replies through the current validators (semantic={args.rescore}); "
            "model behaviour, latency and tokens are from the live run"
        )
        text = json.dumps(
            {"summary": summary, "results": [r.__dict__ for r in results]}, indent=1, default=str
        )
        if args.rescore_out:
            args.rescore_out.write_text(text + "\n")
        print(json.dumps(summary, indent=1))
        return 0
    base = LexicalSupportJudge()
    nli = CrossEncoderNLI(args.device)
    judges: dict[str, SupportJudge] = {
        "lexical": base,
        "rules": GroundedSupportJudge(base, SemanticGuard(None)),
        "rules+nli(default)": GroundedSupportJudge(base, SemanticGuard(nli)),
    }
    if args.config_from.exists():
        chosen = SemanticConfig(**json.loads(args.config_from.read_text())["chosen_config"])
        judges["rules+nli(dev-calibrated)"] = GroundedSupportJudge(base, SemanticGuard(nli, chosen))
    rows = replay(raws, questions, section_retriever, stats, judges)
    out = json.dumps(rows, indent=1) + "\n"
    if args.out:
        args.out.write_text(out)
    for r in rows:
        v = r["verdicts"]
        short = " | ".join(
            f"{k}:{(x['label'] + '/' + x['semantic']) if x else 'no-evidence'}"
            for k, x in v.items()
        )
        print(
            f"{r['id']:4s} {r['claim_kind']:9s} {short}\n      {r['claim'][:150]}", file=sys.stdout
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
