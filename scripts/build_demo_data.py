"""Build ``frontend/lib/demo-data.json`` for the static recorded demo from recorded model runs (no API calls).

    python scripts/build_demo_data.py

For each example in ``data/demo/spec.yaml`` the recorded raw model reply is replayed through the *current* pipeline (retrieval,
evidence, gate, validators, semantic check) and serialised exactly as ``POST /api/v1/debug`` would. The example is published
only if the replay reproduces what was recorded (same status and same cited sections); otherwise the build fails: a demo must
never show an answer the system did not give.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from neural_debugger.backend.api.debug import to_response
from neural_debugger.rag import PipelineConfig, RAGPipeline
from neural_debugger.rag.coverage_eval import build_pipeline
from neural_debugger.rag.providers.testing import ScriptedProvider

SOURCES = {
    "demo": {
        "raw": [
            "data/eval/results/m5_3/demo_recording/live_gemini_gemini-3.1-flash-lite.raw.json",
            "data/eval/results/m5_3/demo_recording/rerun_infra_errors/live_gemini_gemini-3.1-flash-lite.raw.json",
        ],
        "questions": "data/eval/rag/demo_examples.yaml",
    },
    "vd": {
        "raw": [
            "data/eval/results/m5_3/verbose_dev_after/live_gemini_gemini-3.1-flash-lite.raw.json"
        ],
        "questions": "data/eval/rag/verbose_dev.yaml",
    },
}


def _load(source: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cfg = SOURCES[source]
    raw: dict[str, dict[str, Any]] = {}
    for f in cfg["raw"]:
        for item in json.loads(Path(f).read_text()):
            if item["id"] not in raw or raw[item["id"]]["result"]["status"] == "error":
                raw[item["id"]] = item
    questions = {
        q["id"]: q for q in yaml.safe_load(Path(cfg["questions"]).read_text())["questions"]
    }
    return raw, questions


def main() -> int:
    spec = yaml.safe_load(Path("data/demo/spec.yaml").read_text())
    pipe = build_pipeline(
        ["data/corpus", "data/corpus_supplement"],
        "data/embeddings_rag",
        "hybrid-gte-tuned",
        "cpu",
        "coverage",
    )
    loaded = {s: _load(s) for s in SOURCES if Path(SOURCES[s]["raw"][0]).exists()}
    out: list[dict[str, Any]] = []
    for ex in spec["examples"]:
        if ex["source"] not in loaded:
            print(f"skip {ex['id']}: no recording for source {ex['source']!r} yet", file=sys.stderr)
            continue
        raw, questions = loaded[ex["source"]]
        item, question = raw[ex["qid"]], questions[ex["qid"]]["q"]
        replies = [
            c["reply"] for c in item["calls"] if "reply" in c and c.get("purpose") != "probe"
        ]
        # the deployed defaults: coverage profile with semantic rules (what the live recordings were produced with)
        replay = RAGPipeline(
            pipe.retriever,
            ScriptedProvider(replies or ["{}"]),
            PipelineConfig.coverage_profile(semantic="rules"),
            stats=pipe.stats,
        )
        ans = replay.answer(question, None)
        recorded = item["result"]
        same_status = ans.status == recorded["status"] or (
            recorded["status"] in ("answered", "partial") and ans.status in ("answered", "partial")
        )
        same_sections = set(ans.cited_section_ids()) <= set(recorded["cited_sections"]) | set()
        if not (same_status and same_sections):
            print(
                f"FAIL {ex['id']}: replay {ans.status}/{ans.cited_section_ids()} != recorded {recorded['status']}/{recorded['cited_sections']}",
                file=sys.stderr,
            )
            return 1
        resp = to_response(ans, f"demo-{ex['id']}", False).model_dump(mode="json")
        resp["provider"] = "openai_compatible" if replies and ans.provider else None
        resp["model"] = "gemini-3.1-flash-lite" if resp["provider"] else None
        resp["timings"] = {
            "retrieval_ms": resp["timings"]["retrieval_ms"],
            "generation_ms": float(recorded["generation_ms"]),
            "validation_ms": resp["timings"]["validation_ms"],
            "total_ms": float(recorded["total_ms"]),
        }
        out.append(
            {
                "id": ex["id"],
                "title": ex["title"],
                "framework": ex["framework"],
                "question": question.strip(),
                "note": ex["note"],
                "response": resp,
            }
        )
        print(
            f"ok   {ex['id']:14s} {ans.status:10s} confidence={ans.confidence:6s} cited={len(ans.citations)}"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    doc = {
        "meta": {
            **spec["meta"],
            "pipeline_commit": commit,
            "note": "replayed recorded model replies; see docs/demo.md",
        },
        "examples": out,
    }
    Path("frontend/lib/demo-data.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"wrote {len(out)} examples to frontend/lib/demo-data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
