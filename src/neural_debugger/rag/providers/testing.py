"""Deterministic providers for tests and offline evaluation. **Not** language models.

* ``ExtractiveProvider`` reads the evidence blocks out of the prompt and assembles a schema-conforming
  answer from the sentences that overlap most with the question, citing the block each came from. It can
  never hallucinate, so metrics measured with it say nothing about LLM answer quality; they exercise
  retrieval, prompting, parsing, citation validation and latency plumbing.
* ``ScriptedProvider`` replays canned replies/errors in order and records every request, for failure-path
  and attack tests (a "hijacked model" is just a scripted reply that obeys an injected instruction).

The production factory refuses both in ``environment=production``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence

from ..text import term_set
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderError

_BLOCK = re.compile(r'<evidence id="(E\d+)"[^>]*>\n(.*?)\n</evidence>', re.S)
_QUESTION = re.compile(r"<user_question>\n(.*?)\n</user_question>", re.S)
_FENCE = re.compile(r"```([\w+-]*)\n(.*?)```", re.S)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])")


def _body(block: str) -> str:
    return block.split("\n---\n", 1)[-1]


class ExtractiveProvider:
    name = "extractive"
    model = "extractive-v1"

    def generate(self, request: LLMRequest) -> LLMResponse:
        user = request.messages[0].content
        if request.purpose == "judge":
            return LLMResponse(json.dumps(self._judge(user)), self.model)
        q = _QUESTION.search(user)
        question = q.group(1) if q else ""
        blocks = _BLOCK.findall(user)
        q_terms = term_set(question)
        ranked: list[tuple[float, int, str, str]] = []
        for bi, (eid, block) in enumerate(blocks):
            prose = _FENCE.sub(" ", _body(block))
            for sent in _SENT.split(" ".join(prose.split())):
                words = sent.split()
                if not 5 <= len(words) <= 60:
                    continue
                overlap = len(term_set(sent) & q_terms)
                ranked.append((overlap - 0.01 * bi, bi, eid, sent.strip()))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        if not blocks or not ranked or ranked[0][0] < 1:
            return LLMResponse(
                json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "missing_information": ["No relevant excerpt."],
                    }
                ),
                self.model,
            )
        diag = ranked[0]
        steps = [r for r in ranked[1:] if r[2] != diag[2] or r[3] != diag[3]][:3]
        out: dict[str, object] = {
            "status": "answer",
            "diagnosis": {"text": diag[3], "citations": [diag[2]]},
            "steps": [{"text": s[3], "citations": [s[2]]} for s in steps],
        }
        for eid, block in blocks:
            fence = _FENCE.search(_body(block))
            if fence and fence.group(2).strip():
                out["code_suggestions"] = [
                    {
                        "language": fence.group(1) or "python",
                        "code": fence.group(2).strip()[:1500],
                        "explanation": "Example from the cited documentation.",
                        "citations": [eid],
                    }
                ]
                break
        notes = [
            ln[len("version note: ") :]
            for _, b in blocks
            for ln in b.splitlines()
            if ln.startswith("version note: ")
        ]
        if notes:
            out["caveats"] = notes[:2]
        return LLMResponse(
            json.dumps(out), self.model, {"input_tokens": len(user) // 4, "output_tokens": 200}
        )

    @staticmethod
    def _judge(user: str) -> dict[str, str]:
        m = re.search(r"<claim>\n(.*?)\n</claim>", user, re.S)
        claim = m.group(1) if m else ""
        passages = " ".join(re.findall(r"<passage n=\"\d+\">\n(.*?)\n</passage>", user, re.S))
        c = term_set(claim)
        cov = len(c & term_set(passages)) / len(c) if c else 0.0
        verdict = "supported" if cov >= 0.6 else "partial" if cov >= 0.35 else "unsupported"
        return {"verdict": verdict, "reason": f"coverage {cov:.2f}"}


Reply = str | ProviderError | Callable[[LLMRequest], str]


class ScriptedProvider:
    name = "scripted"
    model = "scripted-v1"

    def __init__(self, replies: Sequence[Reply], repeat_last: bool = True) -> None:
        self._replies = list(replies)
        self._repeat_last = repeat_last
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        i = len(self.requests) - 1
        if i >= len(self._replies):
            if not self._repeat_last or not self._replies:
                raise ProviderError("empty", "scripted provider has no more replies")
            i = len(self._replies) - 1
        reply = self._replies[i]
        if isinstance(reply, ProviderError):
            raise reply
        text = reply(request) if callable(reply) else reply
        return LLMResponse(text, self.model, {"input_tokens": 0, "output_tokens": 0})


class UnconfiguredProvider:
    """Stands in when no LLM is configured, so the pipeline degrades instead of failing at start-up."""

    name = "unconfigured"
    model = "none"

    def __init__(self, reason: str = "no LLM provider is configured") -> None:
        self.reason = reason

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise ProviderError("not_configured", self.reason)


class RecordingProvider:
    """Wraps a provider and keeps every request/response pair (for raw-output preservation)."""

    def __init__(self, inner: "LLMProvider") -> None:  # noqa: UP037
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.calls: list[dict[str, object]] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        entry: dict[str, object] = {
            "purpose": request.purpose,
            "system": request.system,
            "messages": [m.content for m in request.messages],
        }
        started = time.perf_counter()
        try:
            resp = self.inner.generate(request)
        except ProviderError as exc:
            entry |= {
                "error": {"kind": exc.kind, "status": exc.status_code, "message": str(exc)[:300]}
            }
            self.calls.append(entry)
            raise
        entry |= {
            "reply": resp.text,
            "usage": resp.usage,
            "stop_reason": resp.stop_reason,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
        self.calls.append(entry)
        return resp
