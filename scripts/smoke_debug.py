"""Smoke-test a running Neural Debugger backend (used by the CI Docker job and for local container checks).

    python scripts/smoke_debug.py http://127.0.0.1:8000                      # a provider is configured: expects an answer
    python scripts/smoke_debug.py http://127.0.0.1:8000 --expect-unconfigured # no provider: expects an informative 503

Standard library only, so it runs on the CI host without installing the project. It checks behaviour that only
shows up when the retrieval stack, corpus, embeddings and model artifacts are really present in the deployment:
the answer must cite documentation sections that exist in the indexed corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

QUESTION = "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"


def call(
    url: str, body: dict[str, object] | None = None, timeout: float = 180.0
) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def wait_healthy(base: str, seconds: float = 90.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            status, body = call(f"{base}/health", timeout=3)
            # /health also checks the database. These smoke runs deliberately have none (the debug endpoint does
            # not need it), so a well-formed "degraded" 503 still proves the application itself is serving.
            if status == 200 or (status == 503 and "database" in body.get("checks", {})):
                print(f"health: HTTP {status} ({body.get('status')})")
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise SystemExit(f"FAIL: {base}/health not healthy after {seconds:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="e.g. http://127.0.0.1:8000")
    ap.add_argument("--expect-unconfigured", action="store_true")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    wait_healthy(base)

    started = time.time()
    status, body = call(f"{base}/api/v1/debug", {"question": QUESTION, "framework": "pytorch"})
    print(
        f"POST /api/v1/debug -> {status} in {time.time() - started:.1f}s (first call loads the retriever)"
    )

    if args.expect_unconfigured:
        err = body.get("error", {})
        assert status == 503, f"expected 503 without a provider, got {status}: {body}"
        assert err.get("code") == "llm_not_configured", err
        assert "ND_LLM_PROVIDER" in err.get("message", ""), (
            "the error must tell the operator what to set"
        )
        sources = (err.get("details") or [{}])[0].get("sources", [])
        assert sources, (
            "retrieval should still have found documentation (proves corpus + model are in the image)"
        )
        print(f"OK informative 503; retrieval found {len(sources)} sources")
        return 0

    assert status == 200, f"expected 200, got {status}: {body}"
    assert body["status"] in ("answered", "partial"), body["status"]
    assert body["citations"], "an answer must cite documentation"
    assert all(c["url"].startswith("https://") for c in body["citations"])
    assert any(c["framework"] == "pytorch" for c in body["citations"]), "should cite PyTorch docs"
    assert body["validation"]["invalid_citations"] == [], body["validation"]
    print(
        f"OK status={body['status']} confidence={body['confidence']} citations={len(body['citations'])} "
        f"provider={body['provider']} total_ms={body['timings']['total_ms']:.0f}"
    )
    assert body["sources"] and all(src["excerpt"] for src in body["sources"]), (
        "sources must carry the text shown"
    )

    # the Milestone 5.2 supplement (official 3.12 'what's new') must be reachable inside the deployment
    status, body2 = call(
        f"{base}/api/v1/debug",
        {
            "question": "Was the distutils package removed from the standard library in Python 3.12?",
            "framework": "python",
        },
    )
    assert status == 200, f"expected 200, got {status}: {body2}"
    urls = [src["url"] for src in body2["sources"]]
    assert any("whatsnew/3.12" in u for u in urls), f"supplement corpus not served; sources: {urls}"
    print("OK supplement corpus reachable (whatsnew/3.12 among the sources)")

    # feedback: accepted, validated, and the question is not stored unless opted in
    status, fb = call(
        f"{base}/api/v1/feedback",
        {
            "request_id": body["request_id"],
            "rating": "not_helpful",
            "reasons": ["incomplete"],
            "question": "private",
        },
    )
    assert status == 201 and fb["stored_question"] is False, (status, fb)
    status, bad = call(f"{base}/api/v1/feedback", {"request_id": "x", "rating": "meh"})
    assert status == 422 and bad["error"]["code"] == "validation_error", (status, bad)
    print("OK feedback endpoint (stored without the question; invalid input rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
