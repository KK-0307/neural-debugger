"""Generate the Milestone 5.3 datasets with a model that never sees the pipeline (free tier, Gemini).

    python scripts/gen_reliability_sets.py dev     # section-seeded verbose questions: DEVELOPMENT, used for diagnosis/tuning
    python scripts/gen_reliability_sets.py final   # free-form questions: FROZEN before any change, never used for tuning

``dev``: sections are sampled from the indexed documentation (seeded, stratified by framework) and Gemini writes a verbose,
realistic question a developer who has *not* read the page would ask, whose answer is in that section. The gold section is
therefore known by construction, which lets the evidence gate be diagnosed on questions that are answerable. It is **model
generated, not human-verified**: a generated question may be answered only partly by its section, and seeding from the page makes
vocabulary overlap easier than real questions, so absolute numbers flatter the system; relative before/after behaviour is the
point. ``final``: free-form questions with no gold (like the Milestone 5.2 held-out set) using different prompts; any question
too similar to one already used anywhere is dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

import yaml

from neural_debugger.data.corpus import load_corpus_overlay
from neural_debugger.rag.live_acceptance import FREE_GEMINI_MODELS, GEMINI_BASE_URL
from neural_debugger.rag.providers.base import (
    LLMMessage,
    LLMRequest,
    ProviderError,
    RequestGovernor,
    RetryPolicy,
)
from neural_debugger.rag.providers.http import OpenAICompatibleProvider
from neural_debugger.rag.text import term_set

SEED = 20261028
USED_FILES = [
    "data/eval/rag/heldout_gemini.yaml",
    "data/eval/rag/independent_gemini.yaml",
    "data/eval/rag/live_acceptance.yaml",
    "data/eval/rag/traceback_dev.yaml",
]
SYSTEM_DEV = (
    "You write realistic questions that developers paste into a debugging assistant. Reply with JSON only: "
    '{"questions": ["...", ...]} with exactly one question per excerpt, in the same order. Use \\n inside JSON '
    "strings for line breaks."
)
STYLE_PROSE = (
    "Write a VERBOSE conversational question of 3 to 6 sentences: the symptom in the developer's own words, what they "
    "already tried, their environment or versions, and a concrete ask. Do not quote the excerpt, do not reuse its headings "
    "and do not mention documentation."
)
STYLE_TRACE = (
    "Start with a realistic pasted traceback or log excerpt of 8 to 20 lines (several library frames with plausible file "
    "paths, ending in the exception or error line), then 1 or 2 sentences of context. Do not quote the excerpt or mention "
    "documentation."
)


def _provider(max_requests: int) -> tuple[OpenAICompatibleProvider, RequestGovernor]:
    gov = RequestGovernor(7.0, max_requests)
    prov = OpenAICompatibleProvider(
        "gemini-3.1-flash-lite",
        GEMINI_BASE_URL,
        "GEMINI_API_KEY",
        retry=RetryPolicy(max_retries=3, max_delay_s=40),
        governor=gov,
    )
    return prov, gov


def _ask(prov: OpenAICompatibleProvider, system: str, prompt: str, temperature: float) -> list[str]:
    text = prov.generate(
        LLMRequest(system, (LLMMessage("user", prompt),), max_tokens=4000, temperature=temperature)
    ).text
    m = re.search(r"\{.*\}", text, re.S)
    items = json.loads(m.group(0))["questions"] if m else []
    return ["\n".join(ln.rstrip() for ln in str(q).strip().splitlines()) for q in items]


def _used_questions() -> list[set[str]]:
    out: list[set[str]] = []
    for f in USED_FILES:
        for q in yaml.safe_load(Path(f).read_text())["questions"]:
            out.append(term_set(q.get("q", "")))
    return out


def _too_similar(q: str, used: list[set[str]], threshold: float = 0.6) -> bool:
    ts = term_set(q)
    return any(len(ts & u) / max(1, len(ts | u)) >= threshold for u in used)


def _sample_sections(per_fw: int) -> list[dict[str, str]]:
    corpus = load_corpus_overlay(["data/corpus", "data/corpus_supplement"])
    rng = random.Random(SEED)
    by_fw: dict[str, list[str]] = {}
    for sid in sorted(corpus.section_ids):
        text = corpus.section_text(sid)
        heading = " ".join(corpus.section_heading_path(sid)).lower()
        alpha = sum(c.isalpha() or c == " " for c in text) / max(1, len(text))
        if not (500 <= len(text) <= 1400) or alpha < 0.82 or "```" in text:
            continue
        if any(
            w in heading
            for w in ("parameters", "returns", "class reference", "summary", "whatsnew")
        ):
            continue
        by_fw.setdefault(sid.split("/", 1)[0], []).append(sid)
    out = []
    for fw in ("python", "pytorch", "fastapi"):
        pool = by_fw[fw]
        rng.shuffle(pool)
        seen_docs: set[str] = set()
        for sid in pool:
            doc = sid.split("#", 1)[0]
            if doc in seen_docs:
                continue
            seen_docs.add(doc)
            out.append(
                {
                    "section": sid,
                    "fw": fw,
                    "title": " > ".join(corpus.section_heading_path(sid)),
                    "text": corpus.section_text(sid),
                }
            )
            if len(seen_docs) >= per_fw:
                break
    return out


def gen_dev(out: Path) -> int:
    prov, gov = _provider(40)
    sections = _sample_sections(12)
    questions: list[dict[str, object]] = []
    for i in range(0, len(sections), 3):
        batch = sections[i : i + 3]
        style = STYLE_TRACE if (i // 3) % 2 else STYLE_PROSE
        parts = [
            f"EXCERPT {n} (from the {s['fw']} documentation, page: {s['title']}):\n{s['text']}"
            for n, s in enumerate(batch, 1)
        ]
        prompt = "\n\n".join(parts) + f"\n\nFor each excerpt write one question. {style}"
        try:
            qs = _ask(prov, SYSTEM_DEV, prompt, 0.7)
        except ProviderError as exc:
            print("generation failed:", exc.kind, file=sys.stderr)
            return 1
        for s, q in zip(batch, qs, strict=False):
            n = len(questions) + 1
            questions.append(
                {
                    "id": f"vd{n}",
                    "kind": "answer",
                    "fw": s["fw"],
                    "style": "traceback" if style is STYLE_TRACE else "prose",
                    "gold_section": s["section"],
                    "q": q,
                }
            )
        print(f"batch {i // 3 + 1}: {len(qs)} questions")
    oos_prompts = [
        (
            "prose",
            STYLE_PROSE,
            "problems in Kubernetes, Rust, React, PostgreSQL, Terraform, Kafka, Go, AWS IAM or Git",
        ),
        (
            "traceback",
            STYLE_TRACE,
            "problems in Django, Flask, NumPy, pandas, scikit-learn, TensorFlow or SQLAlchemy",
        ),
    ]
    for style_name, style, topic in oos_prompts:
        prompt = (
            f"Write 6 different questions about {topic} (NOT Python itself, PyTorch or FastAPI). Reply with the same JSON "
            f"shape. {style}"
        )
        for q in _ask(prov, SYSTEM_DEV, prompt, 0.8)[:6]:
            questions.append(
                {
                    "id": f"vo{len(questions) + 1}",
                    "kind": "abstain",
                    "fw": "other",
                    "style": style_name,
                    "q": q,
                }
            )
    return _write(
        out,
        questions,
        gov.used,
        "DEVELOPMENT (section-seeded; gold section by construction; model-generated)",
    )


def gen_final(out: Path) -> int:
    if out.exists():
        print(f"refusing to overwrite {out}: generated once", file=sys.stderr)
        return 2
    prov, gov = _provider(30)
    used = _used_questions()
    system = (
        "You are a developer who just hit a bug and is typing it into a debugging assistant. Reply with JSON only: "
        '{"questions": ["..."]}. Each item is exactly what the developer would paste or type, in their own voice, specific '
        "and varied. Use \\n inside JSON strings for line breaks."
    )
    batches = [
        (
            "python",
            "answer",
            "prose",
            "Python 3.12 problems (stdlib, typing, asyncio, packaging, files, exceptions, performance). Each item is 3-6 sentences with context and what was tried.",
            8,
        ),
        (
            "python",
            "answer",
            "traceback",
            "Python 3.12 problems. Each item is a realistic pasted traceback of 8-20 lines with several library frames, then 1-2 sentences of context.",
            6,
        ),
        (
            "pytorch",
            "answer",
            "prose",
            "PyTorch 2.x problems (tensors, autograd, nn.Module, optimizers, DataLoader, CUDA, saving/loading, mixed precision). Each item is 3-6 sentences with context and what was tried.",
            8,
        ),
        (
            "pytorch",
            "answer",
            "traceback",
            "PyTorch 2.x problems. Each item is a realistic pasted traceback of 8-20 lines with several torch frames, then 1-2 sentences of context.",
            6,
        ),
        (
            "fastapi",
            "answer",
            "prose",
            "FastAPI problems (routing, Pydantic models, dependencies, background tasks, middleware, auth, testing, deployment). Each item is 3-6 sentences with context and what was tried.",
            8,
        ),
        (
            "fastapi",
            "answer",
            "traceback",
            "FastAPI problems. Each item is a realistic pasted server log or traceback of 8-20 lines with uvicorn/starlette frames, then 1-2 sentences of context.",
            6,
        ),
        (
            "other",
            "abstain",
            "prose",
            "problems in technologies that are NOT Python, PyTorch or FastAPI (Kubernetes, Rust, React, PostgreSQL, Terraform, Kafka, Go, AWS, Git, Docker). 3-6 sentences each.",
            6,
        ),
        (
            "other",
            "abstain",
            "adjacent",
            "problems in Python-ecosystem libraries that are NOT Python itself, PyTorch or FastAPI (Django, Flask, NumPy, pandas, scikit-learn, TensorFlow, SQLAlchemy); some pasted tracebacks.",
            6,
        ),
    ]
    prefix = {"python": "fp", "pytorch": "fy", "fastapi": "ff", "other": "fo"}
    questions: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    dropped = 0
    prompts = []
    for fw, kind, style, topic, n in batches:
        prompt = f"Write {n + 2} different items about: {topic}"
        prompts.append(prompt)
        try:
            items = _ask(prov, system, prompt, 0.9)
        except ProviderError as exc:
            print("generation failed:", exc.kind, file=sys.stderr)
            return 1
        kept = 0
        for q in items:
            if kept >= n:
                break
            if _too_similar(q, used):
                dropped += 1
                continue
            used.append(term_set(q))
            counts[fw] = counts.get(fw, 0) + 1
            kept += 1
            questions.append(
                {"id": f"{prefix[fw]}{counts[fw]}", "kind": kind, "fw": fw, "style": style, "q": q}
            )
        print(f"{fw}/{style}: {kept}")
    print(f"dropped as near-duplicates of earlier sets: {dropped}")
    return _write(
        out,
        questions,
        gov.used,
        "FINAL EVALUATION (free-form; frozen before any Milestone 5.3 change; never used for tuning)",
        prompts,
    )


SYSTEM_CLAIMS = (
    "You write the kind of short statements a debugging assistant gives a developer. Reply with JSON only: "
    '{"claims": [["claim 1 for excerpt 1", "claim 2 for excerpt 1"], ["..."]]} with exactly two claims per excerpt, in order.'
)


def gen_claims(out: Path) -> int:
    """Paraphrased assistant claims, each with its true source section and a *nearby different* section.

    The pairing (claim, true section) should be accepted by a support check; (claim, adjacent section) is a valid citation
    id that does not actually back the claim and should not be accepted. The adjacent section is the dense-nearest other
    section of the same framework whose text lacks the claim's distinctive terms (checked, not assumed).
    """
    from neural_debugger.ml.retrieval.systems import SystemFactory, load_neural_config
    from neural_debugger.rag.sufficiency import TermStats

    corpus = load_corpus_overlay(["data/corpus", "data/corpus_supplement"])
    fac = SystemFactory(
        corpus,
        load_neural_config("configs/neural.yaml", "configs/m4.yaml"),
        device="cpu",
        embedding_dir=Path("data/embeddings_rag"),
    )
    dense = fac.build("dense-gte-small")
    stats = TermStats.from_texts(c.index_text() for c in corpus.chunks)
    rng = random.Random(SEED + 1)
    sections = _sample_sections(30)
    rng.shuffle(sections)
    prov, gov = _provider(40)
    rows: list[dict[str, object]] = []
    for i in range(0, len(sections), 3):
        batch = sections[i : i + 3]
        parts = [
            f"EXCERPT {n} ({s['fw']} documentation, page: {s['title']}):\n{s['text']}"
            for n, s in enumerate(batch, 1)
        ]
        prompt = (
            "\n\n".join(parts)
            + "\n\nFor each excerpt write two different one-sentence statements (a diagnosis or a recommended fix) that follow "
            "from it, in your own words, concise and generic enough to sound like assistant advice. Do not copy sentences."
        )
        try:
            text = prov.generate(
                LLMRequest(
                    SYSTEM_CLAIMS, (LLMMessage("user", prompt),), max_tokens=3000, temperature=0.6
                )
            ).text
            m = re.search(r"\{.*\}", text, re.S)
            groups = json.loads(m.group(0))["claims"] if m else []
        except (ProviderError, json.JSONDecodeError, KeyError) as exc:
            print("batch failed:", type(exc).__name__, file=sys.stderr)
            continue
        for s, claims in zip(batch, groups, strict=False):
            for claim in claims[:2]:
                claim = " ".join(str(claim).split())
                cterms = {
                    t for t in term_set(claim) if stats.idf(t) >= 4.0
                }  # distinctive terms of the claim
                if len(cterms) < 2:
                    continue
                qv = dense.encode_query(claim)
                partner = None
                for h in dense.search_vector(qv, 25, s["fw"]):
                    if h.section_id == s["section"]:
                        continue
                    other = term_set(
                        corpus.section_text(h.section_id)
                        + " "
                        + " ".join(corpus.section_heading_path(h.section_id))
                    )
                    missing = len(cterms - other) / len(cterms)
                    if missing >= 0.5:  # the partner genuinely lacks most of what the claim says
                        partner = h.section_id
                        break
                if partner:
                    rows.append(
                        {
                            "id": f"cr{len(rows) + 1}",
                            "claim": claim,
                            "true_section": s["section"],
                            "adjacent_section": partner,
                            "fw": s["fw"],
                            "distinctive_terms": sorted(cterms),
                        }
                    )
        print(f"batch {i // 3 + 1}: total pairs {len(rows)}")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} claim pairs to {out} ({gov.used} requests)")
    return 0


def _write(
    out: Path,
    questions: list[dict[str, object]],
    requests: int,
    note: str,
    prompts: list[str] | None = None,
) -> int:
    doc = {
        "provenance": {
            "generated_by": "gemini-3.1-flash-lite",
            "date": date.today().isoformat(),
            "seed": SEED,
            "prompt_sha256": hashlib.sha256(
                "\n".join([SYSTEM_DEV, STYLE_PROSE, STYLE_TRACE, *(prompts or [])]).encode()
            ).hexdigest(),
            "note": note + "; model-generated, not human-written or human-verified",
            "requests": requests,
        },
        "questions": questions,
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
    print(f"wrote {len(questions)} questions to {out} ({requests} requests)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=["dev", "final", "claims"])
    args = ap.parse_args()
    assert "gemini-3.1-flash-lite" in FREE_GEMINI_MODELS
    if args.which == "dev":
        return gen_dev(Path("data/eval/rag/verbose_dev.yaml"))
    if args.which == "claims":
        return gen_claims(Path("data/eval/rag/citation_relevance_dev.jsonl"))
    return gen_final(Path("data/eval/rag/final_eval_gemini.yaml"))


if __name__ == "__main__":
    sys.exit(main())
