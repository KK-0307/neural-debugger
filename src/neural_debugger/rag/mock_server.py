"""A local OpenAI-compatible chat server backed by the deterministic ExtractiveProvider.

    python -m neural_debugger.rag.mock_server --port 8765

For harness and deployment smoke tests only (no network, no cost). **It is not a language model**, so nothing
measured against it says anything about real-model behaviour.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .providers.base import LLMMessage, LLMRequest
from .providers.testing import ExtractiveProvider

_PROVIDER = ExtractiveProvider()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"data": [{"id": "mock-extractive"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "not found"}})
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        msgs = [m for m in body["messages"] if m["role"] != "system"]
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        purpose = "judge" if "fact-checking judge" in system else "answer"
        resp = _PROVIDER.generate(
            LLMRequest(
                system, tuple(LLMMessage(m["role"], m["content"]) for m in msgs), purpose=purpose
            )
        )
        self._send(
            200,
            {
                "model": "mock-extractive",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": resp.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": resp.usage.get("input_tokens", 0),
                    "completion_tokens": resp.usage.get("output_tokens", 0),
                },
            },
        )

    def log_message(self, *args: object) -> None:  # silence request logging
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
