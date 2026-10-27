"""Generate the Milestone 5.2 held-out evaluation questions with a model that never saw the corpus or the pipeline.

Unlike ``gen_independent_questions.py`` (whose 30 questions were used to *diagnose* failures and are therefore development
data from Milestone 5.2 on), this set is generated once, hashed and frozen **before any retrieval, gate or normalisation
change**, and is never used for tuning. Half of the in-scope questions embed a pasted traceback or console output so that
stack-trace handling is exercised; the out-of-scope questions include adjacent ecosystems (Flask, NumPy, Django...).

    python scripts/gen_heldout_questions.py --model gemini-3.1-flash-lite

Model-generated, not human-written; no gold answers. Free tier only: same allowlist, spacing and hard request cap as
``live_acceptance``.
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
    "You are a developer who has just hit a bug and is pasting it into a debugging assistant. Reply with JSON only: "
    '{"questions": ["...", ...]}. Each item is exactly what the developer would type or paste, in their own voice. '
    "Make them specific and varied (different libraries, APIs and error classes). Use \\n inside JSON strings for line breaks."
)
PREFIX = {"python": "hp", "pytorch": "ht", "fastapi": "hf", "other": "ho"}
BATCHES = [
    (
        "python",
        "answer",
        "symptom",
        "Python 3.12 problems (standard library, language semantics, packaging, asyncio, typing, files, "
        "exceptions). Describe the symptom in 1-3 sentences, no traceback.",
        6,
    ),
    (
        "python",
        "answer",
        "traceback",
        "Python 3.12 problems. Each item is a realistic pasted traceback (3-8 lines ending in the exception line, "
        "with plausible file names) optionally followed by one short sentence of context.",
        6,
    ),
    (
        "pytorch",
        "answer",
        "symptom",
        "PyTorch 2.x problems (tensors, autograd, nn.Module, optimizers, DataLoader, CUDA, serialization, "
        "distributed). Describe the symptom in 1-3 sentences, no traceback.",
        6,
    ),
    (
        "pytorch",
        "answer",
        "traceback",
        "PyTorch 2.x problems. Each item is a realistic pasted traceback (3-8 lines ending in the exception line, "
        "with plausible file names and torch internals) optionally followed by one short sentence of context.",
        6,
    ),
    (
        "fastapi",
        "answer",
        "symptom",
        "FastAPI problems (routing, Pydantic models, dependencies, background tasks, middleware, security, "
        "testing, deployment). Describe the symptom in 1-3 sentences, no traceback.",
        6,
    ),
    (
        "fastapi",
        "answer",
        "traceback",
        "FastAPI problems. Each item is a realistic pasted traceback or server log excerpt (3-8 lines ending in "
        "the exception or error line) optionally followed by one short sentence of context.",
        6,
    ),
    (
        "other",
        "abstain",
        "symptom",
        "problems in technologies that are NOT Python, PyTorch or FastAPI: Kubernetes, Rust, React, PostgreSQL, "
        "Terraform, Kafka, Go, Docker networking, AWS IAM, Git. Describe the symptom in 1-3 sentences.",
        6,
    ),
    (
        "other",
        "abstain",
        "adjacent",
        "problems in Python-ecosystem libraries that are NOT Python itself, PyTorch or FastAPI: Django, Flask, "
        "NumPy, pandas, scikit-learn, TensorFlow, SQLAlchemy. Some may be pasted tracebacks.",
        6,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--out", type=Path, default=Path("data/eval/rag/heldout_gemini.yaml"))
    ap.add_argument("--pace", type=float, default=7.0)
    args = ap.parse_args()
    if args.model not in FREE_GEMINI_MODELS:
        print(f"refusing: {args.model!r} is not on the free-tier allowlist", file=sys.stderr)
        return 2
    if args.out.exists():
        print(
            f"refusing to overwrite {args.out}: the held-out set is generated once", file=sys.stderr
        )
        return 2
    governor = RequestGovernor(args.pace, max_requests=14)
    provider = OpenAICompatibleProvider(
        args.model,
        GEMINI_BASE_URL,
        "GEMINI_API_KEY",
        retry=RetryPolicy(max_retries=3, max_delay_s=40),
        governor=governor,
    )
    questions: list[dict[str, object]] = []
    prompts: list[str] = []
    counts: dict[str, int] = {}
    for fw, kind, style, topic, n in BATCHES:
        prompt = f"Write {n} different items about: {topic}"
        prompts.append(prompt)
        try:
            text = provider.generate(
                LLMRequest(SYSTEM, (LLMMessage("user", prompt),), max_tokens=3000, temperature=0.8)
            ).text
        except ProviderError as exc:
            print(f"generation failed for {fw}/{style}: {exc.kind}", file=sys.stderr)
            return 1
        m = re.search(r"\{.*\}", text, re.S)
        items = json.loads(m.group(0))["questions"] if m else []
        for q in items[:n]:
            counts[fw] = counts.get(fw, 0) + 1
            questions.append(
                {
                    "id": f"{PREFIX[fw]}{counts[fw]}",
                    "kind": kind,
                    "fw": fw,
                    "style": style,
                    "q": "\n".join(ln.rstrip() for ln in str(q).strip().splitlines()),
                }
            )
        print(f"{fw}/{style}: {len(items[:n])}")
    doc = {
        "provenance": {
            "generated_by": args.model,
            "date": date.today().isoformat(),
            "prompt_sha256": hashlib.sha256("\n".join([SYSTEM, *prompts]).encode()).hexdigest(),
            "note": "HELD-OUT: model-generated (not human-written), no gold answers, generated before any Milestone "
            "5.2 change and never used for tuning; the generating model did not see the corpus or the pipeline",
            "requests": governor.used,
        },
        "questions": questions,
    }
    args.out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
    print(f"wrote {len(questions)} questions to {args.out} ({governor.used} requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
