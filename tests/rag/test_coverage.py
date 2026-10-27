"""Milestone 5.2: traceback API hints, the corpus overlay, and the evidence profile that must not loosen the gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_debugger.data.corpus import load_corpus_overlay
from neural_debugger.rag import PipelineConfig, RAGPipeline
from neural_debugger.rag.coverage_eval import diagnose_item, summarize
from neural_debugger.rag.providers.testing import ExtractiveProvider
from neural_debugger.rag.question import extract_api_hints, prepare_question
from neural_debugger.rag.sufficiency import UniformStats
from neural_debugger.rag.types import RetrievedSection
from tests.rag.helpers import FakeRetriever, section

TORCH_TB = """Traceback (most recent call last):
  File "train.py", line 52, in <module>
    loss = criterion(outputs, labels)
  File "/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/venv/lib/python3.12/site-packages/torch/nn/modules/loss.py", line 1293, in forward
    return F.cross_entropy(input, target)
  File "/venv/lib/python3.12/site-packages/torch/nn/functional.py", line 3479, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target)
IndexError: Target 10 is out of bounds."""


def test_library_frames_become_dotted_api_names_innermost_first() -> None:
    assert extract_api_hints(TORCH_TB) == ("torch.nn.functional.cross_entropy",)
    stdlib = (
        'Traceback (most recent call last):\n  File "a.py", line 3, in <module>\n    json.loads(s)\n'
        '  File "/usr/lib/python3.12/json/__init__.py", line 346, in loads\n    return d.decode(s)\n'
        '  File "/usr/lib/python3.12/json/decoder.py", line 355, in raw_decode\n    raise X\n'
        "json.decoder.JSONDecodeError: Expecting value"
    )
    assert extract_api_hints(stdlib) == ("json.decoder.raw_decode", "json.loads")


def test_user_frames_private_helpers_and_plumbing_give_no_hints() -> None:
    user_only = 'Traceback (most recent call last):\n  File "/app/main.py", line 4, in fib\n    return fib(n-1)\nRecursionError: x'
    assert extract_api_hints(user_only) == ()
    private = (
        'Traceback (most recent call last):\n  File "/v/lib/python3.12/site-packages/torch/_tensor.py", line 581, in backward\n    x\n'
        '  File "/v/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 134, in forward\n    y\n'
    )
    assert extract_api_hints(private) == ()  # private module and a generic "forward" frame
    assert extract_api_hints("no traceback here") == ()


def test_hints_steer_retrieval_but_are_excluded_from_the_gate_text() -> None:
    q = prepare_question(TORCH_TB)
    assert "torch.nn.functional.cross_entropy" in q.retrieval_query
    assert "cross_entropy" not in q.gate_text and "IndexError" in q.gate_text
    assert q.error_lines[-1] == "IndexError: Target 10 is out of bounds."


def test_server_log_lines_do_not_pollute_the_query() -> None:
    q = prepare_question(
        'INFO:     127.0.0.1:52344 - "POST /items/ HTTP/1.1" 500 Internal Server Error\n'
        "ERROR:    Exception in ASGI application\n"
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 18, in create_item\n    return await db.execute(item)\n'
        "sqlalchemy.exc.OperationalError: database is locked"
    )
    assert "127.0.0.1" not in q.retrieval_query and "OperationalError" in q.retrieval_query


# ------------------------------------------------------------------------------------------------ overlay


def _write_corpus(root: Path, rows: list[tuple[str, str]]) -> Path:
    root.mkdir(parents=True)
    chunks, docs = [], []
    for i, (sid, text) in enumerate(rows):
        doc = sid.split("#")[0]
        chunks.append(
            {
                "chunk_id": f"{sid}:0",
                "content": text,
                "content_hash": f"h{i}",
                "doc_id": doc,
                "framework": "python",
                "heading_path": "['T']",
                "ingested_at": "2026-01-01",
                "n_words": "9",
                "ordinal": "0",
                "section_heading": "T",
                "section_id": sid,
                "source_url": "https://x.test/" + doc,
                "title": "T",
                "url": "https://x.test/" + sid,
                "version": "3.12",
            }
        )
        docs.append({"doc_id": doc})
    (root / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks))
    (root / "documents.jsonl").write_text("")
    return root


def test_overlay_merges_corpora_without_touching_either_and_refuses_clashes(tmp_path: Path) -> None:
    base = _write_corpus(
        tmp_path / "base", [("python/3.12/a#x", "alpha beta gamma delta epsilon zeta eta theta")]
    )
    extra = _write_corpus(
        tmp_path / "extra", [("python/3.12/b#y", "iota kappa lambda mu nu xi omicron pi")]
    )
    before = (base / "chunks.jsonl").read_bytes()
    merged = load_corpus_overlay([base, extra])
    assert merged.section_ids == {"python/3.12/a#x", "python/3.12/b#y"}
    assert (base / "chunks.jsonl").read_bytes() == before
    assert merged.sha256 != load_corpus_overlay([base]).sha256
    clash = _write_corpus(
        tmp_path / "clash", [("python/3.12/a#x", "redefined section text goes here today")]
    )
    with pytest.raises(ValueError, match="redefines"):
        load_corpus_overlay([base, clash])


# ------------------------------------------------------------------------ the gate must not loosen


def _sections() -> list[RetrievedSection]:
    # six sections with distinct vocabulary (the deduplicator would merge look-alikes) and one that holds the answer
    out = [
        section(
            i,
            " ".join(
                f"{w}{i}"
                for w in (
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "epsilon",
                    "zeta",
                    "eta",
                    "theta",
                    "iota",
                )
            )
            + f" remarks section number {i} about something unrelated entirely.",
            doc=f"pytorch/2.14/doc{i}",
        )
        for i in range(1, 7)
    ]
    out.append(
        section(
            7,
            "The zorblax flag controls the quuxification pipeline stage. " * 3,
            doc="pytorch/2.14/doc7",
        )
    )
    return out


def test_coverage_profile_shows_more_evidence_but_keeps_gate_decisions_unchanged() -> None:
    question = (
        "How does the zorblax flag change the quuxification pipeline stage in my training loop?"
    )
    sections = _sections()

    def run(cfg: PipelineConfig) -> tuple[int, str]:
        pipe = RAGPipeline(FakeRetriever(sections), ExtractiveProvider(), cfg, stats=UniformStats())
        prep = pipe.prepare(question)
        return len(prep.evidence), prep.sufficiency.level

    n_m5, gate_m5 = run(PipelineConfig())
    n_cov, gate_cov = run(PipelineConfig.coverage_profile())
    assert n_m5 == 5 and n_cov == 7  # the model is shown two more sections ...
    assert (
        gate_m5 == gate_cov == "none"
    )  # ... but the sentence-7 match does not rescue an out-of-reach question

    # control: if the gate were allowed to see the larger set, the extra sections WOULD lift coverage and pass it
    loose = PipelineConfig.coverage_profile(gate_evidence=None)
    assert run(loose)[1] != "none"


def test_diagnose_item_classifies_the_stage_that_lost_the_answer() -> None:
    sections = _sections()
    pipe = RAGPipeline(
        FakeRetriever(sections),
        ExtractiveProvider(),
        PipelineConfig.coverage_profile(),
        stats=UniformStats(),
    )
    pipe.retriever._known = {s.section_id for s in sections}  # type: ignore[attr-defined]

    def stage(question: str, gold_section: str, key: str) -> str:
        item = {
            "id": "x",
            "kind": "answer",
            "q": question,
            "gold": [{"section": gold_section, "key": key}],
        }
        return str(diagnose_item(pipe, item)["stage"])

    covered = "alpha1 beta1 gamma1 delta1 epsilon1 remarks section"
    assert stage(covered, sections[0].section_id, "alpha1") == "reachable"
    # shown to the model (7th section) but outside what the gate judges, so the gate would abstain first
    assert (
        stage("zorblax quuxification pipeline stage flag", sections[6].section_id, "zorblax")
        == "gate_blocked"
    )
    assert stage(covered, sections[0].section_id, "not-in-the-text") == "text_truncated"
    assert stage(covered, "pytorch/2.14/nowhere#x", "x") == "corpus_gap"
    rows = [
        diagnose_item(
            pipe,
            {
                "id": "a",
                "kind": "answer",
                "q": covered,
                "gold": [{"section": sections[0].section_id, "key": "alpha1"}],
            },
        )
    ]
    assert summarize(rows)["gold_reachable"] == 1
