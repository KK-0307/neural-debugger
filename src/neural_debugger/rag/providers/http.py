"""Real LLM providers over HTTPS (``httpx``): Anthropic Messages API and OpenAI-compatible chat API.

Credentials come from the environment only (the *name* of the variable is configuration, the value is
never stored in settings objects, logged, or included in ``repr``). The OpenAI-compatible provider also
covers locally hosted servers (Ollama, vLLM, llama.cpp server), which need no key and cost nothing.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

import httpx

from .base import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    RetryPolicy,
    with_retries,
)

ANTHROPIC_VERSION = "2023-06-01"


def _classify_status(resp: httpx.Response) -> ProviderError:
    status = resp.status_code
    try:
        msg = str(resp.json().get("error", {}).get("message", ""))[:300]
    except Exception:  # body is not JSON
        msg = resp.text[:200]
    retry_after: float | None = None
    with contextlib.suppress(ValueError):
        retry_after = float(resp.headers.get("retry-after", ""))
    text = f"HTTP {status}: {msg}".strip()
    if status in (401, 403):
        return ProviderError("auth", text, status_code=status)
    if status == 429:
        return ProviderError(
            "rate_limit", text, retryable=True, status_code=status, retry_after=retry_after
        )
    if status in (408, 409) or status >= 500:  # 529 = Anthropic "overloaded"
        return ProviderError(
            "server", text, retryable=True, status_code=status, retry_after=retry_after
        )
    return ProviderError("bad_request", text, status_code=status)


class _HTTPProvider:
    name = "http"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str | None,
        client: httpx.Client | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self._client = client or httpx.Client()
        self._retry = retry or RetryPolicy()

    def __repr__(self) -> str:  # never include credentials
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    def _api_key(self, required: bool) -> str | None:
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if required and not key:
            raise ProviderError(
                "not_configured", f"environment variable {self.api_key_env} is not set"
            )
        return key

    def _post(
        self, path: str, headers: dict[str, str], body: dict[str, Any], timeout: float
    ) -> Any:
        try:
            resp = self._client.post(
                self.base_url + path, headers=headers, json=body, timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "timeout", f"request timed out after {timeout:.0f}s", retryable=True
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError("network", f"{type(exc).__name__}", retryable=True) from exc
        if resp.status_code >= 400:
            raise _classify_status(resp)
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError("server", "response body was not JSON", retryable=True) from exc

    def _timed(self, request: LLMRequest, call: Any) -> LLMResponse:
        started = time.perf_counter()
        resp: LLMResponse = with_retries(lambda: call(request), self._retry)
        return LLMResponse(
            resp.text,
            resp.model,
            resp.usage,
            resp.stop_reason,
            (time.perf_counter() - started) * 1000,
        )


class AnthropicProvider(_HTTPProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        client: httpx.Client | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        super().__init__(model, base_url, api_key_env, client, retry)

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self._timed(request, self._once)

    def _once(self, request: LLMRequest) -> LLMResponse:
        key = self._api_key(required=True)
        assert key is not None
        body = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = self._post("/v1/messages", headers, body, request.timeout_s)
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text.strip():
            raise ProviderError("empty", "the model returned no text", retryable=False)
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=str(data.get("model", self.model)),
            usage={
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            },
            stop_reason=data.get("stop_reason"),
        )


class OpenAICompatibleProvider(_HTTPProvider):
    """``POST {base_url}/chat/completions``; works with OpenAI, Ollama (``/v1``), vLLM, llama.cpp."""

    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str | None = "OPENAI_API_KEY",
        client: httpx.Client | None = None,
        retry: RetryPolicy | None = None,
        send_json_mode: bool = False,
        key_required: bool = True,
    ) -> None:
        super().__init__(model, base_url, api_key_env, client, retry)
        self.send_json_mode = send_json_mode
        self.key_required = key_required

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self._timed(request, self._once)

    def _once(self, request: LLMRequest) -> LLMResponse:
        key = self._api_key(required=self.key_required)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": "system", "content": request.system},
                *({"role": m.role, "content": m.content} for m in request.messages),
            ],
        }
        if request.json_mode and self.send_json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        data = self._post("/chat/completions", headers, body, request.timeout_s)
        choices = data.get("choices") or []
        text = (choices[0].get("message", {}).get("content") if choices else "") or ""
        if not text.strip():
            raise ProviderError("empty", "the model returned no text", retryable=False)
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=str(data.get("model", self.model)),
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
            stop_reason=choices[0].get("finish_reason") if choices else None,
        )
