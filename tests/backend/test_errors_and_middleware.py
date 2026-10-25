import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from neural_debugger.backend.errors import AppError
from neural_debugger.backend.logging_config import JsonFormatter


class Payload(BaseModel):
    error_text: str = Field(min_length=1, max_length=50)


@pytest.fixture
def app_with_probe_routes(client: TestClient) -> TestClient:
    """Adds probe routes to the real app so error paths can be exercised end to end."""
    app: FastAPI = client.app  # type: ignore[assignment]

    @app.post("/_probe/validate")
    def validate(p: Payload) -> dict[str, str]:
        return {"ok": p.error_text}

    @app.get("/_probe/app-error")
    def app_error() -> None:
        raise AppError("insufficient_evidence", "Not enough evidence.", status_code=409)

    @app.get("/_probe/boom")
    def boom() -> None:
        raise RuntimeError("secret: Traceback with user data")

    return client


def test_unknown_route_returns_structured_404(client: TestClient) -> None:
    r = client.get("/nope")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "http_404"
    assert err["request_id"] == r.headers["x-request-id"]


def test_validation_error_is_structured_and_does_not_echo_input(
    app_with_probe_routes: TestClient,
) -> None:
    secret = "RuntimeError: private stack trace " * 5  # > max_length
    r = app_with_probe_routes.post("/_probe/validate", json={"error_text": secret})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"][0]["loc"] == ["body", "error_text"]
    assert "private stack trace" not in r.text


def test_app_error_maps_to_status_and_code(app_with_probe_routes: TestClient) -> None:
    r = app_with_probe_routes.get("/_probe/app-error")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "insufficient_evidence"


def test_unhandled_exception_returns_generic_500(app_with_probe_routes: TestClient) -> None:
    r = app_with_probe_routes.get("/_probe/boom")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text
    assert r.headers["x-request-id"]


def test_request_id_is_echoed_or_generated(client: TestClient) -> None:
    assert client.get("/health", headers={"X-Request-ID": "abc123"}).headers["x-request-id"] == (
        "abc123"
    )
    generated = client.get("/health").headers["x-request-id"]
    assert len(generated) == 32


def test_access_log_has_no_query_string_or_body(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="neural_debugger.access"):
        client.get("/health?token=SECRET_VALUE")
    rec = next(r for r in caplog.records if r.name == "neural_debugger.access")
    assert rec.__dict__["path"] == "/health"
    assert "SECRET_VALUE" not in json.dumps(rec.__dict__, default=str)


def test_json_formatter_emits_valid_json_without_traceback_text() -> None:
    try:
        raise ValueError("user secret")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, "f", 1, "oops", (), sys.exc_info())
    record.request_id = "r1"
    out = json.loads(JsonFormatter().format(record))
    assert out["event"] == "oops"
    assert out["request_id"] == "r1"
    assert out["exc_type"] == "ValueError"
    assert "user secret" not in json.dumps(out)
