from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neural_debugger.backend.config import Settings
from neural_debugger.backend.main import create_app
from neural_debugger.backend.services import feedback as feedback_module
from tests.rag.helpers import QUESTION, answer_json, pipeline


@pytest.fixture
def client(settings: Settings, tmp_path: Path) -> Iterator[TestClient]:
    settings = settings.model_copy(update={"feedback_dir": str(tmp_path / "fb")})
    with TestClient(create_app(settings, pipeline(replies=[answer_json()])[0])) as c:
        yield c


def stored(client: TestClient) -> list[dict[str, object]]:
    return client.app.state.feedback.read_all()  # type: ignore[no-any-return,attr-defined]


def test_feedback_is_stored_without_the_question_unless_the_user_opts_in(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/v1/feedback",
        json={
            "request_id": "req-1",
            "rating": "not_helpful",
            "reasons": ["wrong_source", "incomplete", "wrong_source"],
            "comment": "  the cited page is about CPU tensors  ",
            "answer_status": "answered",
            "confidence": "medium",
            "question": "my private traceback",
        },
    )
    assert r.status_code == 201 and r.json()["stored_question"] is False
    [rec] = stored(client)
    assert rec["question"] is None and rec["reasons"] == ["incomplete", "wrong_source"]
    assert rec["comment"] == "the cited page is about CPU tensors"
    assert rec["request_id"] == "req-1" and rec["rating"] == "not_helpful"

    r2 = client.post(
        "/api/v1/feedback",
        json={
            "request_id": "req-2",
            "rating": "helpful",
            "include_question": True,
            "question": "KeyError: 'a'",
        },
    )
    assert r2.status_code == 201 and r2.json()["stored_question"] is True
    assert stored(client)[1]["question"] == "KeyError: 'a'"


def test_secrets_in_free_text_are_redacted_before_they_reach_disk(client: TestClient) -> None:
    fake = "AI" + "za" + "x" * 30
    client.post(
        "/api/v1/feedback",
        json={
            "request_id": "r",
            "rating": "not_helpful",
            "comment": f"I pasted my key {fake} and api_key=abcdef123456 by mistake",
            "include_question": True,
            "question": f"Authorization: Bearer abcdefghijklmnop1234 {fake}",
        },
    )
    text = client.app.state.feedback.path.read_text()  # type: ignore[attr-defined]
    assert fake not in text and "abcdef123456" not in text and "abcdefghijklmnop1234" not in text
    assert text.count("[REDACTED]") >= 3


@pytest.mark.parametrize(
    "body",
    [
        {"request_id": "", "rating": "helpful"},
        {"request_id": "bad id with spaces", "rating": "helpful"},
        {"request_id": "r", "rating": "meh"},
        {"request_id": "r", "rating": "helpful", "reasons": ["not-a-reason"]},
        {"request_id": "r", "rating": "helpful", "comment": "x" * 1001},
    ],
)
def test_invalid_feedback_is_rejected_with_a_structured_error(
    client: TestClient, body: dict[str, object]
) -> None:
    r = client.post("/api/v1/feedback", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert stored(client) == []


def test_a_full_store_answers_503_and_does_not_affect_the_rest_of_the_api(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feedback_module, "MAX_FILE_BYTES", 10)
    ok = {"request_id": "r", "rating": "helpful"}
    assert (
        client.post("/api/v1/feedback", json=ok).status_code == 201
    )  # first write may exceed: file did not exist
    r = client.post("/api/v1/feedback", json=ok)
    assert r.status_code == 503 and r.json()["error"]["code"] == "feedback_storage_full"
    assert client.post("/api/v1/debug", json={"question": QUESTION}).status_code == 200


def test_debug_response_exposes_what_the_ui_needs_to_explain_uncertainty(
    client: TestClient,
) -> None:
    body = client.post(
        "/api/v1/debug", json={"question": QUESTION}, headers={"X-Request-ID": "req-9"}
    ).json()
    assert body["request_id"] == "req-9"
    assert (
        body["evidence_gate"]["level"] in ("ok", "weak")
        and 0 < body["evidence_gate"]["coverage"] <= 1
    )
    src = body["sources"][0]
    assert src["excerpt"] and src["url"].startswith("https://")
    assert {"semantic", "semantic_reason", "support"} <= set(body["diagnosis"])
    assert {"contradicted_claims", "disputed_claims", "unverified_claims"} <= set(
        body["validation"]
    )
