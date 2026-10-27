"""Provider interface for the generation layer.

The pipeline depends only on ``LLMProvider.generate``; swapping vendors is a configuration change.
Providers raise ``ProviderError`` with a stable ``kind`` so the pipeline can degrade predictably
(return the retrieved sources with an explicit "no generated answer" status) instead of crashing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

ErrorKind = Literal[
    "not_configured",
    "auth",
    "rate_limit",
    "timeout",
    "network",
    "server",
    "bad_request",
    "empty",
    "budget_exhausted",  # local guard: the configured request cap was reached (nothing was sent)
]


class ProviderError(Exception):
    """A generation failure. ``retryable`` tells wrappers whether another attempt is sensible."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: tuple[LLMMessage, ...]
    max_tokens: int = 1500
    temperature: float = 0.0
    json_mode: bool = True  # hint; providers without native JSON mode rely on the prompt
    timeout_s: float = 60.0
    purpose: str = "answer"  # "answer" | "judge" | "repair": lets test providers route behaviour


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    latency_ms: float = 0.0


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    sleep: Callable[[float], None] = time.sleep

    def delay(self, attempt: int, error: ProviderError) -> float:
        if error.retry_after is not None:
            return min(error.retry_after, self.max_delay_s)
        return min(self.base_delay_s * (2.0**attempt), self.max_delay_s)


def with_retries(call: Callable[[], LLMResponse], policy: RetryPolicy) -> LLMResponse:
    attempt = 0
    while True:
        try:
            return call()
        except ProviderError as exc:
            if not exc.retryable or attempt >= policy.max_retries:
                raise
            policy.sleep(policy.delay(attempt, exc))
            attempt += 1


@dataclass
class RequestGovernor:
    """Client-side guard so a free-tier key is never hammered: spacing between calls and a hard request cap.

    ``max_requests`` counts calls *attempted* (including retries); when it is reached further calls raise
    ``ProviderError("budget_exhausted")`` without touching the network. There is no fallback to any other
    model or provider.
    """

    min_interval_s: float = 0.0
    max_requests: int | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    used: int = 0
    _last: float | None = field(default=None, repr=False)

    def before_request(self) -> None:
        if self.max_requests is not None and self.used >= self.max_requests:
            raise ProviderError(
                "budget_exhausted",
                f"request cap of {self.max_requests} reached; no request was sent",
            )
        if self._last is not None and self.min_interval_s > 0:
            wait = self.min_interval_s - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self.used += 1
        self._last = self.clock()
