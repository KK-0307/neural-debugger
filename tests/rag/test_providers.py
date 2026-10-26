"""HTTP providers against a mocked transport: request shape, error mapping, retries, credential hygiene."""

import json
import logging

import httpx
import pytest

from neural_debugger.rag.providers.base import LLMMessage, LLMRequest, ProviderError, RetryPolicy
from neural_debugger.rag.providers.http import AnthropicProvider, OpenAICompatibleProvider

REQ = LLMRequest(system="sys", messages=(LLMMessage("user", "hi"),), max_tokens=50)
SECRET = "sk-test-SECRET-123"


def client(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def fast(n: int = 2) -> RetryPolicy:
    return RetryPolicy(max_retries=n, sleep=lambda s: sleeps.append(s))


sleeps: list[float] = []


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps.clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)


def test_anthropic_request_shape_and_response_parsing() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "m-1",
                "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
                "usage": {"input_tokens": 11, "output_tokens": 3},
                "stop_reason": "end_turn",
            },
        )

    p = AnthropicProvider("m-1", client=client(handler), retry=fast())
    r = p.generate(REQ)
    assert r.text == "hello world" and r.usage == {"input_tokens": 11, "output_tokens": 3}
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == SECRET and seen["headers"]["anthropic-version"]  # type: ignore[index]
    body = seen["body"]
    assert body["system"] == "sys" and body["messages"] == [{"role": "user", "content": "hi"}]  # type: ignore[index]
    assert SECRET not in repr(p)


def test_missing_key_is_reported_without_a_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    called = []
    p = AnthropicProvider("m", client=client(lambda r: called.append(r) or httpx.Response(200)))
    with pytest.raises(ProviderError) as e:
        p.generate(REQ)
    assert e.value.kind == "not_configured" and not called and "ANTHROPIC_API_KEY" in str(e.value)


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (401, "auth", False),
        (403, "auth", False),
        (429, "rate_limit", True),
        (500, "server", True),
        (529, "server", True),
        (400, "bad_request", False),
    ],
)
def test_http_errors_are_classified(status: int, kind: str, retryable: bool) -> None:
    p = AnthropicProvider(
        "m",
        client=client(lambda r: httpx.Response(status, json={"error": {"message": "x"}})),
        retry=fast(0),
    )
    with pytest.raises(ProviderError) as e:
        p.generate(REQ)
    assert (e.value.kind, e.value.retryable, e.value.status_code) == (kind, retryable, status)


def test_retries_transient_errors_with_backoff_then_succeeds() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(
                429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}}
            )
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    r = AnthropicProvider("m", client=client(handler), retry=fast(2)).generate(REQ)
    assert r.text == "ok" and len(calls) == 3 and sleeps == [2.0, 2.0]


def test_gives_up_after_max_retries_and_does_not_retry_auth() -> None:
    n = []
    p = AnthropicProvider(
        "m", client=client(lambda r: n.append(1) or httpx.Response(503, json={})), retry=fast(2)
    )
    with pytest.raises(ProviderError):
        p.generate(REQ)
    assert len(n) == 3
    n.clear()
    p = AnthropicProvider(
        "m", client=client(lambda r: n.append(1) or httpx.Response(401, json={})), retry=fast(2)
    )
    with pytest.raises(ProviderError):
        p.generate(REQ)
    assert len(n) == 1


def test_timeout_and_network_errors_map_to_provider_errors() -> None:
    def boom(exc: Exception):  # type: ignore[no-untyped-def]
        def handler(request: httpx.Request) -> httpx.Response:
            raise exc

        return handler

    with pytest.raises(ProviderError) as e:
        AnthropicProvider("m", client=client(boom(httpx.ReadTimeout("t"))), retry=fast(0)).generate(
            REQ
        )
    assert e.value.kind == "timeout" and e.value.retryable
    with pytest.raises(ProviderError) as e:
        AnthropicProvider(
            "m", client=client(boom(httpx.ConnectError("c"))), retry=fast(0)
        ).generate(REQ)
    assert e.value.kind == "network"


def test_empty_and_non_json_replies() -> None:
    with pytest.raises(ProviderError) as e:
        AnthropicProvider(
            "m", client=client(lambda r: httpx.Response(200, json={"content": []}))
        ).generate(REQ)
    assert e.value.kind == "empty"
    with pytest.raises(ProviderError) as e:
        AnthropicProvider(
            "m", client=client(lambda r: httpx.Response(200, text="<html>")), retry=fast(0)
        ).generate(REQ)
    assert e.value.kind == "server"


def test_openai_compatible_shape_local_server_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["h"] = dict(request.headers)
        seen["b"] = json.loads(request.content)
        seen["u"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    p = OpenAICompatibleProvider(
        "llama",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        key_required=False,
        client=client(handler),
        send_json_mode=True,
    )
    r = p.generate(REQ)
    assert (
        seen["u"] == "http://localhost:11434/v1/chat/completions"
        and "authorization" not in seen["h"]
    )  # type: ignore[operator]
    assert seen["b"]["messages"][0] == {"role": "system", "content": "sys"}  # type: ignore[index]
    assert seen["b"]["response_format"] == {"type": "json_object"}  # type: ignore[index]
    assert r.usage == {"input_tokens": 5, "output_tokens": 2}
    with pytest.raises(ProviderError) as e:
        OpenAICompatibleProvider("m", client=client(handler)).generate(REQ)
    assert e.value.kind == "not_configured"


def test_credentials_never_appear_in_logs_or_error_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    p = AnthropicProvider(
        "m",
        client=client(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})),
        retry=fast(0),
    )
    with pytest.raises(ProviderError) as e:
        p.generate(REQ)
    assert SECRET not in str(e.value) and SECRET not in caplog.text and SECRET not in repr(p)
