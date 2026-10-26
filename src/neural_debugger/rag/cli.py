"""Ask a question from the command line.

    python -m neural_debugger.rag.cli "RuntimeError: expected scalar type Float but found Double"
    python -m neural_debugger.rag.cli --provider anthropic --model claude-haiku-5-5 "..."   # needs ANTHROPIC_API_KEY
    python -m neural_debugger.rag.cli --provider extractive --retriever bm25 "..."         # offline, no LLM

The default retriever is the production hybrid (``hybrid-gte-tuned``). ``--provider extractive`` is a
deterministic test provider, not a language model.
"""

from __future__ import annotations

import argparse
import json
import sys

from neural_debugger.backend.config import Settings
from neural_debugger.backend.services.rag import build_pipeline

from .types import RAGAnswer


def render(ans: RAGAnswer) -> str:
    out: list[str] = [f"[{ans.status} | confidence: {ans.confidence}]"]
    if ans.abstain_reason:
        out.append(f"reason: {ans.abstain_reason}")
    if ans.diagnosis:
        out += ["", "Diagnosis:", f"  {ans.diagnosis.text}"]
    if ans.steps:
        out += ["", "Steps:"] + [f"  {i}. {s.text}" for i, s in enumerate(ans.steps, 1)]
    for snippet in ans.code_suggestions:
        out += ["", f"Code ({snippet.language}; not executed): {snippet.explanation.text}"]
        out += ["    " + ln for ln in snippet.code.splitlines()]
    for conflict in ans.conflicts:
        out += ["", f"Conflict: {conflict.text}"]
    if ans.caveats:
        out += ["", "Caveats:"] + [f"  - {c}" for c in ans.caveats]
    if ans.missing_information:
        out += ["", "Missing information:"] + [f"  - {m}" for m in ans.missing_information]
    if ans.citations:
        out += ["", "Sources:"] + [
            f"  [{c.id}] {c.title} > {c.heading}\n       {c.url}" for c in ans.citations
        ]
    elif ans.evidence:
        out += ["", "Closest documentation (not cited):"] + [
            f"  - {e.title}: {e.url}" for e in ans.evidence[:3]
        ]
    t = ans.timings
    out.append(
        f"\n(retrieval {t.retrieval_ms:.0f} ms, generation {t.generation_ms:.0f} ms, "
        f"validation {t.validation_ms:.0f} ms)"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question", nargs="?", help="question text (default: read stdin)")
    ap.add_argument(
        "--provider",
        default="none",
        choices=["none", "anthropic", "openai_compatible", "extractive"],
    )
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument(
        "--retriever", default="hybrid-gte-tuned", help="system name from configs/*.yaml"
    )
    ap.add_argument("--framework", choices=["python", "pytorch", "fastapi"])
    ap.add_argument("--support", choices=["off", "lexical", "llm"], default="lexical")
    ap.add_argument("--json", action="store_true", help="print the full structured answer")
    args = ap.parse_args(argv)
    question = args.question or sys.stdin.read()
    s = Settings(
        environment="development",
        llm_provider=args.provider,
        llm_model=args.model,
        llm_base_url=args.base_url,
        retriever_system=args.retriever,
        rag_support_check=args.support,
    )
    if args.api_key_env:
        s.llm_api_key_env = args.api_key_env
    ans = build_pipeline(s).answer(question, args.framework)
    print(json.dumps(ans.as_dict(), indent=1, default=str) if args.json else render(ans))
    return 0 if ans.status in ("answered", "partial", "abstained") else 1


if __name__ == "__main__":
    sys.exit(main())
