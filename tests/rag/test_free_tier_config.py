"""Free-tier safety rails: request cap, spacing, key sourcing, Gemini preset, redaction, model allowlist."""

import json
from pathlib import Path

import httpx
import pytest

from neural_debugger.backend.config import Settings
from neural_debugger.backend.services.rag import GEMINI_BASE_URL, build_provider
from neural_debugger.rag.live_acceptance import (
    FREE_GEMINI_MODELS,
    main,
    preflight,
    redact,
    redact_obj,
)
from neural_debugger.rag.providers.base import (
    LLMMessage,
    LLMRequest,
    ProviderError,
    RequestGovernor,
    RetryPolicy,
)
from neural_debugger.rag.providers.http import OpenAICompatibleProvider

REQ = LLMRequest("sys", (LLMMessage("user", "hi"),))
# assembled at runtime so no key-shaped literal sits in the repository (secret scanners)
KEY = "AI" + "za" + "SyFAKEKEYFORTESTING1234567890abcd"


def ok_client(seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_governor_caps_requests_and_never_touches_the_network_after_the_cap() -> None:
    seen: list[httpx.Request] = []
    p = OpenAICompatibleProvider(
        "m",
        "http://x",
        None,
        client=ok_client(seen),
        key_required=False,
        governor=RequestGovernor(max_requests=2),
    )
    p.generate(REQ)
    p.generate(REQ)
    with pytest.raises(ProviderError) as e:
        p.generate(REQ)
    assert e.value.kind == "budget_exhausted" and len(seen) == 2


def test_governor_enforces_minimum_spacing() -> None:
    now = [0.0]
    slept: list[float] = []
    g = RequestGovernor(
        min_interval_s=6.0,
        clock=lambda: now[0],
        sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s))[0],
    )
    g.before_request()
    now[0] += 2.0
    g.before_request()
    assert slept == [4.0] and g.used == 2


def test_retries_count_against_the_cap() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "quota"}})

    p = OpenAICompatibleProvider(
        "m",
        "http://x",
        None,
        key_required=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry=RetryPolicy(max_retries=5, sleep=lambda s: None),
        governor=RequestGovernor(max_requests=3),
    )
    with pytest.raises(ProviderError) as e:
        p.generate(REQ)
    assert e.value.kind == "budget_exhausted" and len(calls) == 3


def test_key_is_read_from_env_or_dotenv_only_for_its_own_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"OTHER=zzz\n# comment\nexport GEMINI_API_KEY='{KEY}'\n")
    seen: list[httpx.Request] = []
    p = OpenAICompatibleProvider(
        "m", "http://x", "GEMINI_API_KEY", client=ok_client(seen), dotenv_path=str(dotenv)
    )
    p.generate(REQ)
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    import os

    assert "GEMINI_API_KEY" not in os.environ and KEY not in repr(p)
    missing = OpenAICompatibleProvider(
        "m", "http://x", "GEMINI_API_KEY", client=ok_client([]), dotenv_path=str(tmp_path / "nope")
    )
    with pytest.raises(ProviderError) as e:
        missing.generate(REQ)
    assert e.value.kind == "not_configured"


def test_gemini_preset_uses_the_openai_compatible_endpoint_and_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    s = Settings(
        environment="test",
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        llm_reasoning_effort="low",
        llm_min_interval_seconds=7,
        llm_max_requests=10,
    )
    p = build_provider(s)
    assert isinstance(p, OpenAICompatibleProvider) and p.base_url == GEMINI_BASE_URL.rstrip("/")
    assert p.api_key_env == "GEMINI_API_KEY" and p.reasoning_effort == "low"
    assert p.governor.min_interval_s == 7 and p.governor.max_requests == 10 and KEY not in repr(p)
    seen: list[httpx.Request] = []
    p._client = ok_client(seen)  # noqa: SLF001
    p.generate(REQ)
    body = json.loads(seen[0].content)
    assert (
        str(seen[0].url) == GEMINI_BASE_URL + "/chat/completions"
        and body["reasoning_effort"] == "low"
    )
    assert body["model"] == "gemini-2.5-flash"


def test_redaction_removes_the_actual_key_and_key_shaped_strings() -> None:
    text = f"error calling with {KEY} and Authorization: Bearer abcdefghijklmnop123456 and sk-abcdefghijklmnopqrstuv"
    clean = redact(text, [KEY])
    assert (
        KEY not in clean
        and "abcdefghijklmnop123456" not in clean
        and "sk-abcdefghijklmnopqrstuv" not in clean
    )
    assert redact_obj({"a": [KEY, {"b": f"x {KEY}"}]}, [KEY]) == {
        "a": ["[REDACTED]", {"b": "x [REDACTED]"}]
    }


def test_preflight_checks_model_availability_without_generating() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models") and request.method == "GET"
        return httpx.Response(
            200, json={"data": [{"id": "models/gemini-2.5-flash"}, {"id": "models/gemini-2.5-pro"}]}
        )

    transport = httpx.MockTransport(handler)
    real_get = httpx.get
    httpx.get = lambda url, **kw: httpx.Client(transport=transport).get(url, **kw)  # type: ignore[assignment]
    try:
        assert "gemini-2.5-flash" in preflight("http://x/v1beta/openai", "gemini-2.5-flash", KEY)
        with pytest.raises(ProviderError):
            preflight("http://x/v1beta/openai", "gemini-nonexistent", KEY)
    finally:
        httpx.get = real_get


def test_paid_only_models_are_refused_and_missing_key_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    assert "gemini-2.5-pro" not in FREE_GEMINI_MODELS
    assert main(["--provider", "gemini", "--model", "gemini-2.5-pro"]) == 2
    assert "not on the documented free-tier allowlist" in capsys.readouterr().err
    assert main(["--provider", "gemini", "--model", "gemini-2.5-flash"]) == 3
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err and KEY not in err
