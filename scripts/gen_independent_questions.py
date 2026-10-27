"""Generate an evaluation question set with a *different* model than the one that wrote the pipeline.

The 21 live-acceptance questions were written by the same agent that built the RAG pipeline, so they share its
blind spots. This script asks a hosted free-tier model (Gemini) - which never sees the corpus, the pipeline or the
other questions - to write realistic developer questions. They are model-generated, **not human-written** and carry
no gold answers; the provenance (model, date, prompt hash) is stored with them.

    python scripts/gen_independent_questions.py --model gemini-3.1-flash-lite

Uses the free tier only: the same allowlist, spacing and hard request cap as ``live_acceptance``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import yaml

from neural_debugger.rag.live_acceptance import FREE_GEMINI_MODELS, GEMINI_BASE_URL
from neural_debugger.rag.providers.base import (
    LLMMessage,
    LLMRequest,
    ProviderError,
    RequestGovernor,
    RetryPolicy,
)
from neural_debugger.rag.providers.http import OpenAICompatibleProvider

SYSTEM = (
    "You write realistic questions that developers paste into a debugging assistant. Reply with JSON only: "
    '{"questions": ["...", ...]}. Questions are self-contained, 1-3 sentences, and vary in style: some paste a '
    "real error message or traceback line, some describe a symptom, some ask how to do something. No code blocks."
)
PREFIX = {"python": "py", "pytorch": "pt", "fastapi": "fa", "other": "ot"}
BATCHES = [
    ("python", "answer", "Python 3.12 standard library or language errors and how-to questions", 8),
    (
        "pytorch",
        "answer",
        "PyTorch 2.x training, tensor, autograd, CUDA, DataLoader and serialization problems",
        8,
    ),
    (
        "fastapi",
        "answer",
        "FastAPI web API problems: validation, dependencies, CORS, auth, background tasks, testing",
        8,
    ),
    (
        "other",
        "abstain",
        "problems with technologies that are NOT Python, PyTorch or FastAPI (for example Kubernetes, Rust, React, "
        "PostgreSQL, Docker networking, Excel, Git internals)",
        6,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--out", type=Path, default=Path("data/eval/rag/independent_gemini.yaml"))
    ap.add_argument("--pace", type=float, default=7.0)
    args = ap.parse_args()
    if args.model not in FREE_GEMINI_MODELS:
        print(f"refusing: {args.model!r} is not on the free-tier allowlist", file=sys.stderr)
        return 2
    governor = RequestGovernor(args.pace, max_requests=8)
    provider = OpenAICompatibleProvider(
        args.model,
        GEMINI_BASE_URL,
        "GEMINI_API_KEY",
        retry=RetryPolicy(max_retries=2, max_delay_s=30),
        governor=governor,
    )
    questions: list[dict[str, object]] = []
    prompts: list[str] = []
    for fw, kind, topic, n in BATCHES:
        prompt = f"Write {n} different questions about: {topic}."
        prompts.append(prompt)
        try:
            text = provider.generate(
                LLMRequest(SYSTEM, (LLMMessage("user", prompt),), max_tokens=1500, temperature=0.7)
            ).text
        except ProviderError as exc:
            print(f"generation failed for {fw}: {exc.kind}", file=sys.stderr)
            return 1
        m = re.search(r"\{.*\}", text, re.S)
        items = json.loads(m.group(0))["questions"] if m else []
        for i, q in enumerate(items[:n], 1):
            questions.append(
                {"id": f"{PREFIX[fw]}{i}", "kind": kind, "fw": fw, "q": " ".join(str(q).split())}
            )
        print(f"{fw}: {len(items[:n])} questions")
    doc = {
        "provenance": {
            "generated_by": args.model,
            "date": date.today().isoformat(),
            "prompt_sha256": hashlib.sha256("\n".join([SYSTEM, *prompts]).encode()).hexdigest(),
            "note": "model-generated questions; not human-written, no gold answers; the generating model "
            "did not see the corpus, the pipeline or any other question",
            "requests": governor.used,
        },
        "questions": questions,
    }
    args.out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
    print(f"wrote {len(questions)} questions to {args.out} ({governor.used} requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
