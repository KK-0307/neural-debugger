import json
import threading
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from neural_debugger.backend.config import Settings
from neural_debugger.backend.main import create_app
from neural_debugger.rag import RAGPipeline
from neural_debugger.rag.providers.base import ProviderError
from neural_debugger.rag.providers.testing import ScriptedProvider, UnconfiguredProvider
from tests.rag.helpers import QUESTION, FakeRetriever, answer_json, pipeline, section


def make_client(settings: Settings, pipe: RAGPipeline) -> Iterator[TestClient]:
    with TestClient(create_app(settings, pipe)) as c:
        yield c


@pytest.fixture
def ok_client(settings: Settings) -> Iterator[TestClient]:
    yield from make_client(settings, pipeline(replies=[answer_json()])[0])


def test_debug_returns_a_structured_cited_answer(ok_client: TestClient) -> None:
    r = ok_client.post(
        "/api/v1/debug", json={"question": QUESTION}, headers={"X-Request-ID": "req-1"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == "req-1" and body["status"] == "answered"
    assert body["diagnosis"]["citations"] == ["E1"] and body["citations"][0]["url"].startswith(
        "https://"
    )
    assert any(s["cited"] for s in body["sources"]) and body["trace"] is None
    assert set(body["timings"]) == {"retrieval_ms", "generation_ms", "validation_ms", "total_ms"}
    assert ok_client.post(
        "/api/v1/debug", json={"question": QUESTION, "include_trace": True}
    ).json()["trace"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "x"},
        {"question": "ok question", "framework": "django"},
        {"question": "q" * 20001},
    ],
)
def test_debug_validates_input_without_echoing_it(
    ok_client: TestClient, payload: dict[str, object]
) -> None:
    r = ok_client.post("/api/v1/debug", json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert "q" * 100 not in r.text


def test_abstention_is_a_200_with_no_citations(settings: Settings) -> None:
    pipe, _, _ = pipeline(sections=[], replies=[answer_json()])
    c = next(make_client(settings, pipe))
    r = c.post("/api/v1/debug", json={"question": "something nobody documented"})
    assert (
        r.status_code == 200 and r.json()["status"] == "abstained" and r.json()["citations"] == []
    )


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (ProviderError("auth", "x"), 503, "llm_unavailable"),
        (ProviderError("timeout", "x", retryable=True), 504, "llm_timeout"),
        (ProviderError("bad_request", "x"), 502, "llm_bad_request"),
    ],
)
def test_provider_failures_map_to_http_errors_that_still_list_sources(
    settings: Settings, error: ProviderError, status: int, code: str
) -> None:
    c = next(make_client(settings, pipeline(replies=[error])[0]))
    r = c.post("/api/v1/debug", json={"question": QUESTION})
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert r.json()["error"]["details"][0]["sources"][0]["url"].startswith("https://")


def test_unconfigured_llm_and_unavailable_retrieval(settings: Settings) -> None:
    c = next(
        make_client(settings, RAGPipeline(FakeRetriever([section(1)]), UnconfiguredProvider()))
    )
    r = c.post("/api/v1/debug", json={"question": QUESTION})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_not_configured"
    c = next(
        make_client(settings, pipeline(sections=RuntimeError("down"), replies=[answer_json()])[0])
    )
    assert (
        c.post("/api/v1/debug", json={"question": QUESTION}).json()["error"]["code"]
        == "retrieval_unavailable"
    )


def test_concurrency_limit_returns_429(settings: Settings) -> None:
    gate = threading.Event()
    release = threading.Event()

    def slow(_req: object) -> str:
        gate.set()
        release.wait(5)
        return answer_json()

    s = settings.model_copy(update={"rag_max_concurrent": 1})
    pipe = RAGPipeline(FakeRetriever([section(1)]), ScriptedProvider([slow]))
    with TestClient(create_app(s, pipe)) as c:
        t = threading.Thread(target=lambda: c.post("/api/v1/debug", json={"question": QUESTION}))
        t.start()
        assert gate.wait(5)
        r = c.post("/api/v1/debug", json={"question": QUESTION})
        release.set()
        t.join()
    assert r.status_code == 429 and r.json()["error"]["code"] == "busy"


def test_response_never_contains_secrets_or_question_in_logs(
    settings: Settings, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-SECRET-123")
    caplog.set_level("DEBUG")
    c = next(make_client(settings, pipeline(replies=[answer_json()])[0]))
    r = c.post("/api/v1/debug", json={"question": QUESTION + " my_private_table_name"})
    assert "sk-test-SECRET-123" not in r.text and "sk-test-SECRET-123" not in caplog.text
    assert "my_private_table_name" not in caplog.text and json.loads(r.text)["status"] == "answered"
