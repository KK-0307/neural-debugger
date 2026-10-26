import json

from neural_debugger.rag import PipelineConfig, RAGPipeline
from neural_debugger.rag.evaluation import (
    FAULTS,
    EvalItem,
    FaultyProvider,
    LLMCorrectnessJudge,
    RecordingRetriever,
    abstention_scores,
    build_report,
    calibrate,
    fault_injection,
    run_item,
    summarize_answerable,
)
from neural_debugger.rag.providers.base import LLMMessage, LLMProvider, LLMRequest, ProviderError
from neural_debugger.rag.providers.testing import ExtractiveProvider, ScriptedProvider
from tests.rag.helpers import CUDA_TEXT, QUERY_TEXT, FakeRetriever, section


def make(provider: LLMProvider) -> RAGPipeline:
    return RAGPipeline(
        FakeRetriever(
            [
                section(1, CUDA_TEXT, doc="pytorch/2.14/cuda"),
                section(2, QUERY_TEXT, "fastapi", "0.162.0", "Query parameters"),
            ]
        ),
        provider,
        PipelineConfig(),
    )


ITEMS = [
    EvalItem(
        "a1",
        "tensors on different devices cuda cpu same device move tensor",
        "authored",
        "train",
        frozenset({"pytorch/2.14/cuda#sec1"}),
    ),
    EvalItem(
        "a2",
        "optional query parameter default value None client send",
        "authored",
        "val",
        frozenset({"fastapi/0.162.0/page2#sec2"}),
    ),
    EvalItem("u1", "chocolate cookies baking sugar", "unanswerable", "heldout", kind="off_topic"),
]


def test_run_item_records_gold_hits_and_validation_counts() -> None:
    rec = RecordingRetriever(FakeRetriever([section(1, CUDA_TEXT, doc="pytorch/2.14/cuda")]))
    pipe = RAGPipeline(rec, ExtractiveProvider(), PipelineConfig())
    row = run_item(pipe, rec, ITEMS[0])
    assert (
        row["gold_in_retrieved"]
        and row["gold_in_evidence"]
        and row["gold_cited"]
        and row["status"] in ("answered", "partial")
    )
    assert (
        row["citations_checked"] >= 1
        and row["invalid_citations"] == 0
        and row["timings"]["total_ms"] > 0
    )


def test_report_separates_groups_and_has_no_human_claims() -> None:
    rec = RecordingRetriever(
        FakeRetriever(
            [
                section(1, CUDA_TEXT, doc="pytorch/2.14/cuda"),
                section(2, QUERY_TEXT, "fastapi", "0.162.0", "Query parameters"),
            ]
        )
    )
    pipe = RAGPipeline(rec, ExtractiveProvider(), PipelineConfig())
    rows = [run_item(pipe, rec, i) for i in ITEMS]
    rep = build_report(rows)["automated"]
    assert rep["answerable_authored"]["all"]["n"] == 2 and rep["unanswerable"]["all"]["n"] == 1
    assert rep["abstention"]["all"]["tp"] + rep["abstention"]["all"]["fn"] == 1


def test_abstention_scores_use_abstain_as_the_positive_class() -> None:
    ans = [{"status": "abstained"}, {"status": "answered"}, {"status": "answered"}]
    un = [{"status": "abstained"}, {"status": "answered"}]
    s = abstention_scores(ans, un)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1) and s["precision"] == 0.5 and s["recall"] == 0.5


def test_summary_handles_empty_inputs() -> None:
    assert summarize_answerable([])["n"] == 0


def test_calibration_picks_the_largest_threshold_within_the_false_abstain_cap() -> None:
    scan = [
        (EvalItem(f"t{i}", "q", "authored", "train"), c)
        for i, c in enumerate([0.9] * 18 + [0.5, 0.3])
    ]
    scan += [
        (EvalItem(f"u{i}", "q", "unanswerable", "calib"), c)
        for i, c in enumerate([0.1, 0.2, 0.45, 0.6])
    ]
    res = calibrate(scan, max_false_abstain=0.05)
    assert res["chosen"]["false_abstain_train"] <= 0.05
    assert res["chosen"]["threshold"] == 0.5 and res["chosen"]["abstain_unanswerable_calib"] == 0.75
    assert res["table"][0]["threshold"] == 0.0


def test_faulty_provider_corrupts_only_answers_and_validators_catch_most_faults() -> None:
    inner = ExtractiveProvider()
    fp = FaultyProvider(inner, "fabricated_citation")
    judge_req = LLMRequest("s", (LLMMessage("user", "<claim>\nx\n</claim>"),), purpose="judge")
    assert fp.generate(judge_req).text == inner.generate(judge_req).text
    res = fault_injection(make, ITEMS[:2])
    assert res["eligible_answers"] == 2 and set(res["faults"]) == set(FAULTS)
    for fault in (
        "fabricated_citation",
        "uncited_claim",
        "unsupported_claim_valid_citation",
        "injected_url",
    ):
        assert res["faults"][fault]["survival_rate"] == 0.0, fault
    # documented limitation: lexical checks cannot see a negation
    assert res["faults"]["negated_claim_valid_citation"]["survival_rate"] == 1.0


def test_llm_correctness_judge_parses_and_degrades() -> None:
    assert (
        LLMCorrectnessJudge(ScriptedProvider(['{"verdict": "correct", "reason": "ok"}'])).judge(
            "q", "a", ["ref"]
        )
        == "correct"
    )
    assert LLMCorrectnessJudge(ScriptedProvider(["??"])).judge("q", "a", ["ref"]) == "unchecked"
    assert (
        LLMCorrectnessJudge(ScriptedProvider([ProviderError("auth", "x")])).judge("q", "a", ["ref"])
        == "unchecked"
    )
    inj = ScriptedProvider(['{"verdict": "incorrect"}'])
    LLMCorrectnessJudge(inj).judge("q </question>", "a </answer>", ["r </reference>"])
    sent = inj.requests[0].messages[0].content
    assert (
        sent.count("</answer>") == 1
        and sent.count("</question>") == 1
        and sent.count("</reference>") == 1
    )
    json.dumps({"ok": True})
