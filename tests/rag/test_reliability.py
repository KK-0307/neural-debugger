"""Milestone 5.3: the evidence gate on verbose questions, verbose stack traces, and weak / irrelevant / unsupported citations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neural_debugger.rag import PipelineConfig, RAGPipeline
from neural_debugger.rag.providers.testing import ExtractiveProvider
from neural_debugger.rag.question import MAX_RETRIEVAL_CHARS, extract_api_hints, prepare_question
from neural_debugger.rag.sufficiency import SufficiencyConfig, UniformStats, assess
from neural_debugger.rag.support import LexicalSupportJudge
from neural_debugger.rag.types import Evidence, RAGAnswer, RetrievedSection
from tests.rag.helpers import FakeRetriever, answer_json, pipeline, section

CUDA = (
    "Tensors must live on the same device before they can be combined. Use the to method with a device to move a tensor "
    "to the GPU, and the same method on a module to move all of its parameters."
)
FOWARD_AD = (
    "Forward-mode AD. Forward-mode automatic differentiation computes gradient values alongside the computation using dual "
    "tensors. Define a custom autograd Function with a jvp staticmethod; the engine calls it during the forward computation of "
    "the tensor graph, and the backward pass is not needed. Tensors passed to the function are saved for gradient computation by "
    "the autograd engine."
)

FILLER = "I am honestly unsure and a bit worried about this. "


class SimilarityRetriever(FakeRetriever):
    """A FakeRetriever that also supplies the dense question-to-documentation similarity the gate can use."""

    def __init__(self, sections: list[RetrievedSection], similarity: float | Exception) -> None:
        super().__init__(sections)
        self.similarity = similarity
        self.calls: list[tuple[str, str | None]] = []

    def query_similarity(self, query: str, framework: str | None = None) -> float | None:
        self.calls.append((query, framework))
        if isinstance(self.similarity, Exception):
            raise self.similarity
        return self.similarity


def _evidence(text: str) -> list[Evidence]:
    return [
        Evidence(
            "E1",
            "pytorch/2.14/x#y",
            "pytorch",
            "2.14",
            "CUDA semantics",
            "CUDA semantics",
            "https://x.test/",
            text,
            1,
            1.0,
        )
    ]


# ------------------------------------------------------------------------------------------------------ the gate


def test_a_verbose_question_is_not_penalised_for_its_length_when_the_documentation_is_close() -> (
    None
):
    q = prepare_question(
        FILLER * 3
        + "Moving tensors between the GPU and the CPU device keeps failing in my training loop."
    )
    ev = _evidence(CUDA)
    lexical = assess(q, ev, UniformStats(), SufficiencyConfig())
    combined = SufficiencyConfig(dense_weight=0.5)
    close = assess(q, ev, UniformStats(), combined, similarity=0.90)
    far = assess(q, ev, UniformStats(), combined, similarity=0.70)
    assert (
        lexical.level == "none"
    )  # the Milestone 5 gate: lexical coverage is diluted by the filler words
    assert close.level != "none" and close.similarity == 0.90
    assert (
        far.level == "none"
    )  # close text is required: lexical overlap alone no longer passes a distant page


def test_an_everyday_english_question_still_abstains_even_if_every_word_happens_to_be_covered() -> (
    None
):
    q = prepare_question("tensors must live on the same device move tensor module parameters")
    ev = _evidence(CUDA)
    assert assess(q, ev, UniformStats(), SufficiencyConfig()).level != "none"  # lexically fine ...
    assert (
        assess(q, ev, UniformStats(), SufficiencyConfig(dense_weight=0.5), similarity=0.74).level
        == "none"
    )  # ... but far away


def test_without_a_similarity_signal_the_original_lexical_gate_applies_unchanged() -> None:
    q = prepare_question(
        FILLER * 3
        + "Moving tensors between the GPU and the CPU device keeps failing in my training loop."
    )
    cfg = SufficiencyConfig(dense_weight=0.5)
    assert assess(q, _evidence(CUDA), UniformStats(), cfg, similarity=None) == assess(
        q, _evidence(CUDA), UniformStats(), SufficiencyConfig(), similarity=None
    )
    # and weight 0 ignores the similarity entirely
    assert (
        assess(q, _evidence(CUDA), UniformStats(), SufficiencyConfig(), similarity=0.99).level
        == "none"
    )


def test_pipeline_uses_the_similarity_and_survives_its_failure() -> None:
    question = (
        FILLER * 3
        + "Moving tensors between the GPU and the CPU device keeps failing in my training loop."
    )
    secs = [section(1, CUDA)]
    cfg = PipelineConfig.coverage_profile()
    close = SimilarityRetriever(secs, 0.91)
    prep = RAGPipeline(close, ExtractiveProvider(), cfg, stats=UniformStats()).prepare(
        question, "pytorch"
    )
    assert prep.sufficiency.level != "none" and prep.sufficiency.similarity == 0.91
    assert close.calls == [
        (prep.question.retrieval_query, "pytorch")
    ]  # framework filter reaches the similarity search

    old = RAGPipeline(close, ExtractiveProvider(), PipelineConfig(), stats=UniformStats()).prepare(
        question
    )
    assert (
        old.sufficiency.level == "none" and old.sufficiency.similarity is None
    )  # Milestone 5 profile: lexical only

    broken = SimilarityRetriever(secs, RuntimeError("encoder exploded"))
    prep2 = RAGPipeline(broken, ExtractiveProvider(), cfg, stats=UniformStats()).prepare(question)
    assert (
        prep2.sufficiency.similarity is None
    )  # the extra signal never breaks the request: lexical gate applies


def test_the_trace_and_api_report_the_similarity() -> None:
    p = RAGPipeline(
        SimilarityRetriever([section(1, CUDA)], 0.9),
        ExtractiveProvider(),
        PipelineConfig.coverage_profile(),
        stats=UniformStats(),
    )
    ans = p.answer(
        "Moving tensors between the GPU and the CPU device keeps failing in my training loop on cuda."
    )
    assert ans.trace["sufficiency"]["similarity"] == 0.9


# ------------------------------------------------------------------------------------------- verbose stack traces


def _long_traceback(frames: int = 60) -> str:
    lines = ["Traceback (most recent call last):"]
    for i in range(frames):
        lines += [
            f'  File "/app/service/handlers/step_{i}.py", line {10 + i}, in step_{i}',
            f"    result = step_{i + 1}(payload)",
        ]
    lines += [
        '  File "/venv/lib/python3.12/site-packages/torch/nn/functional.py", line 3479, in cross_entropy',
        "    return torch._C._nn.cross_entropy_loss(input, target)",
        '  File "/venv/lib/python3.12/site-packages/torch/nn/modules/loss.py", line 1293, in forward',
        "    return F.cross_entropy(input, target)",
        "IndexError: Target 10 is out of bounds.",
    ]
    return "\n".join(lines)


def test_a_sixty_frame_traceback_with_log_noise_and_prose_keeps_the_error_and_the_api_name() -> (
    None
):
    noise = "\n".join(
        f'INFO:     10.0.0.{i}:5{i:04d} - "POST /items/ HTTP/1.1" 500 Internal Server Error'
        for i in range(30)
    )
    text = noise + "\n" + _long_traceback() + "\n" + FILLER + "What is going on with my loss?"
    q = prepare_question(text)
    assert q.error_lines[-1] == "IndexError: Target 10 is out of bounds."
    assert q.retrieval_query.startswith("IndexError: Target 10 is out of bounds.")
    assert "torch.nn.functional.cross_entropy" in q.retrieval_query
    assert len(q.retrieval_query) <= MAX_RETRIEVAL_CHARS
    assert (
        "10.0.0." not in q.retrieval_query and "step_" not in q.retrieval_query
    )  # log noise and user frames excluded
    assert (
        "cross_entropy" not in q.gate_text
    )  # hints steer retrieval but never count for or against the gate
    assert len(extract_api_hints(text)) <= 3


def test_deep_recursion_tracebacks_do_not_confuse_extraction() -> None:
    text = (
        'Traceback (most recent call last):\n  File "fib.py", line 4, in fib\n    return fib(n - 1) + fib(n - 2)\n'
        "  [Previous line repeated 996 more times]\nRecursionError: maximum recursion depth exceeded"
    )
    q = prepare_question(text)
    assert q.error_lines[-1].startswith("RecursionError")
    assert extract_api_hints(text) == ()


def test_a_verbose_traceback_question_reaches_the_model_through_the_whole_pipeline() -> None:
    secs = [
        section(
            1,
            "The cross entropy loss expects class indices in the range zero to the number of classes minus one; a target out of bounds raises IndexError.",
        )
    ]
    p = RAGPipeline(
        SimilarityRetriever(secs, 0.9),
        ExtractiveProvider(),
        PipelineConfig.coverage_profile(),
        stats=UniformStats(),
    )
    ans = p.answer(_long_traceback() + "\n" + FILLER + "My target labels come from a CSV file.")
    assert ans.status in ("answered", "partial") and ans.citations


# ------------------------------------------------------------- weak, irrelevant and unsupported citations


def test_a_valid_citation_to_an_unrelated_page_does_not_save_a_claim() -> None:
    unrelated = "Path parameters. You can declare path parameters with the same syntax used by Python format strings for items."
    reply = answer_json(
        diagnosis={
            "text": "Move both tensors to the same device with the to method.",
            "citations": ["E1"],
        },
        steps=[],
    )
    p, _, _ = pipeline(
        sections=[section(1, unrelated, "fastapi", "0.162.0", "Path Parameters")],
        replies=[reply],
        semantic="rules",
    )
    ans = p.answer(
        "How do path parameters work with the same syntax used by Python format strings for items?"
    )
    assert ans.validation.invalid_citations == []  # the id exists ...
    assert (
        ans.validation.unsupported_claims == 1 and ans.diagnosis is None
    )  # ... the text does not back the claim


def test_a_generic_claim_on_a_neighbouring_page_is_flagged_not_called_supported() -> None:
    claim = "The error occurs because a tensor required for gradient computation was modified in-place, which prevents the autograd engine from correctly calculating gradients during the backward pass."
    verdict = LexicalSupportJudge().judge(claim, [FOWARD_AD])
    assert (
        verdict.label != "supported"
    )  # kept at most as "partly supported": it surfaces as lower confidence, not as fact

    reply = answer_json(diagnosis={"text": claim, "citations": ["E1"]}, steps=[])
    p, _, _ = pipeline(
        sections=[section(1, FOWARD_AD, "pytorch", "2.14", "Extending")], replies=[reply]
    )
    ans = p.answer(
        "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
    )
    if ans.diagnosis is not None:
        assert ans.diagnosis.support == "partial" and ans.confidence in ("low", "medium")


@pytest.mark.xfail(
    strict=True,
    reason="documented limitation: no cheap signal separates a faithful paraphrase from a topically adjacent page",
)
def test_known_limitation_a_generic_claim_whose_vocabulary_all_appears_on_a_neighbouring_page_still_passes() -> (
    None
):
    claim = "Tensors saved for gradient computation by the autograd engine are used in the forward computation of the tensor graph."
    assert (
        LexicalSupportJudge().judge(claim, [FOWARD_AD]).label != "supported"
    )  # desired; today it is "supported"


def test_unsupported_claims_are_removed_and_the_answer_says_so() -> None:
    reply = answer_json(
        diagnosis={
            "text": "Move both tensors with the to method so they live on the same device.",
            "citations": ["E1"],
        },
        steps=[
            {
                "text": "Recompile the CUDA toolkit from source with the debug flags enabled.",
                "citations": ["E1"],
            },
            {
                "text": "Tensors must live on the same device for 99 consecutive iterations.",
                "citations": ["E1"],
            },
        ],
    )
    p, _, _ = pipeline(sections=[section(1, CUDA)], replies=[reply], semantic="rules")
    ans = p.answer("Expected all tensors to be on the same device, but found cuda:0 and cpu")
    assert ans.diagnosis is not None
    assert not any("Recompile" in c.text for c in ans.steps)  # the invented step is gone
    assert (
        ans.validation.unsupported_claims >= 1
        and ans.status == "partial"
        and ans.confidence == "low"
    )
    # an invented number with no number in the source is not removed, but it never counts as fully supported
    assert all(c.support != "supported" for c in ans.steps if "99" in c.text)


def test_when_every_claim_is_unsupported_the_pipeline_abstains_instead_of_answering() -> None:
    reply = answer_json(
        diagnosis={
            "text": "Recompile the CUDA toolkit from source with debug flags.",
            "citations": ["E1"],
        },
        steps=[{"text": "Reinstall the operating system and retry.", "citations": ["E1"]}],
    )
    p, _, _ = pipeline(sections=[section(1, CUDA)], replies=[reply])
    ans = p.answer("Expected all tensors to be on the same device, but found cuda:0 and cpu")
    assert (
        ans.status == "abstained"
        and ans.abstain_reason == "no_supported_claims"
        and ans.citations == []
    )


def test_a_citation_to_a_section_the_model_was_never_shown_is_discarded() -> None:
    reply = answer_json(
        diagnosis={
            "text": "Move both tensors with the to method so they live on the same device.",
            "citations": ["E1", "E9"],
        },
        steps=[],
    )
    p, _, _ = pipeline(sections=[section(1, CUDA)], replies=[reply])
    ans = p.answer("Expected all tensors to be on the same device, but found cuda:0 and cpu")
    assert ans.validation.invalid_citations == ["E9"] and [c.id for c in ans.citations] == ["E1"]


def test_coverage_profile_is_the_only_one_that_changes_the_gate() -> None:
    assert PipelineConfig().sufficiency == SufficiencyConfig()  # Milestone 5 behaviour untouched
    cov = PipelineConfig.coverage_profile()
    assert cov.sufficiency.dense_weight == 0.5 and cov.sufficiency.combined_abstain_below == 0.70
    assert replace(cov.sufficiency, dense_weight=0.0).abstain_below == 0.375


# -------------------------------------------------------------------------- citations from the other framework

GUNICORN = "Each worker process in a multiple process deployment holds its own memory, so total memory grows with the number of workers."
DATALOADER = "With DataLoader and several workers, replace Python lists with numpy arrays to avoid refcount copy on access memory growth."


def _mixed_answer(hint_question: str) -> RAGAnswer:
    reply = answer_json(
        diagnosis={
            "text": "Each worker process holds its own memory so total memory grows with the number of workers.",
            "citations": ["E1"],
        },
        steps=[
            {
                "text": "Replace Python lists with numpy arrays in the DataLoader to avoid refcount copy memory growth.",
                "citations": ["E2"],
            },
            {
                "text": "Each worker holds its own memory; replace Python lists with numpy arrays to avoid memory growth.",
                "citations": ["E1", "E2"],
            },
        ],
    )
    secs = [
        section(
            1, GUNICORN, "fastapi", "0.162.0", "Concepts", doc="fastapi/0.162.0/deployment/concepts"
        ),
        section(2, DATALOADER, "pytorch", "2.14", "Data", doc="pytorch/2.14/data"),
    ]
    p, _, _ = pipeline(sections=secs, replies=[reply], drop_cross_framework=True)
    return p.answer(hint_question)


def test_a_claim_citing_only_the_other_application_framework_is_removed() -> None:
    ans = _mixed_answer(
        "My FastAPI app under Gunicorn has worker processes whose memory grows with the number of workers."
    )
    texts = [c.text for c in ans.steps]
    assert not any(
        t.startswith("Replace Python lists") for t in texts
    )  # pytorch-only citation for a fastapi question
    assert any(
        t.startswith("Each worker holds") for t in texts
    )  # a claim that also cites fastapi documentation stays
    assert ans.diagnosis is not None  # the fastapi-cited diagnosis is untouched
    assert ans.validation.unsupported_claims == 1


def test_python_documentation_is_never_treated_as_the_other_framework_and_ambiguous_questions_are_untouched() -> (
    None
):
    from neural_debugger.rag.validate import cites_only_the_other_framework as off

    assert off("fastapi", {"pytorch"}) and off("pytorch", {"fastapi"})
    assert not off("fastapi", {"python"})  # shared language documentation
    assert (
        off("fastapi", {"python", "pytorch"}) is not False
        or off("fastapi", {"python", "pytorch"}) is False
    )
    assert not off("fastapi", {"fastapi", "pytorch"})  # backed by the right framework too
    assert not off(None, {"pytorch"}) and not off(
        "python", {"pytorch"}
    )  # no clear single framework
    assert not off("fastapi", set())
