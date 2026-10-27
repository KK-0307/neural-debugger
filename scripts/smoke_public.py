"""Verify a deployed Neural Debugger (the single-container full-stack image, or any public URL) from the outside.

    python scripts/smoke_public.py https://<host>                     # site, config, privacy, error handling (no model call)
    python scripts/smoke_public.py https://<host> --answers 3         # also N real answers (Python, PyTorch, FastAPI) through the model
    python scripts/smoke_public.py https://<host> --check-limits      # also exceed the per-minute limit (uses out-of-scope questions,
                                                                      # which the pre-model gate declines: no model quota is spent)

Standard library only. Exits non-zero on the first failed expectation. Prints one line per check, never a secret.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ANSWER_QUESTIONS = [
    (
        "python",
        "RuntimeWarning: coroutine 'main' was never awaited. Why does this appear and how do I fix it?",
    ),
    (
        "pytorch",
        "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!",
    ),
    (
        "fastapi",
        "I POST JSON to my FastAPI endpoint and get 422 Unprocessable Entity with 'Field required' for body. What is wrong?",
    ),
]
OUT_OF_SCOPE = "My Kubernetes pod is stuck in CrashLoopBackOff after I added a readiness probe to the deployment. How do I debug it?"


def call(url: str, body: dict[str, object] | None = None, timeout: float = 240.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def jcall(url: str, body: dict[str, object] | None = None, timeout: float = 240.0):
    status, headers, raw = call(url, body, timeout)
    try:
        return status, headers, json.loads(raw or b"{}")
    except json.JSONDecodeError:
        return status, headers, {"_raw": raw[:200].decode("utf-8", "replace")}


def ok(msg: str) -> None:
    print(f"OK   {msg}", flush=True)


def wait_ready(base: str, seconds: float) -> None:
    """A free host sleeps when idle: the first request wakes it and can take minutes."""
    start, deadline = time.time(), time.time() + seconds
    while time.time() < deadline:
        try:
            status, _, body = jcall(f"{base}/health", timeout=20)
            if status == 200 and body.get("status") == "ok":
                ok(f"health ok after {time.time() - start:.0f}s")
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(3)
    raise SystemExit(f"FAIL: {base}/health not ready after {seconds:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base")
    ap.add_argument("--answers", type=int, default=0, help="real model answers to request (0-3)")
    ap.add_argument("--check-limits", action="store_true")
    ap.add_argument(
        "--wait", type=float, default=420.0, help="seconds to wait for a sleeping host to wake"
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")
    wait_ready(base, args.wait)

    # 1. the real frontend is served from the same origin
    status, headers, raw = call(f"{base}/")
    page = raw.decode("utf-8", "replace")
    assert status == 200 and "Neural Debugger" in page and "<script" in page, (status, page[:200])
    assert (
        headers.get("x-content-type-options", "").lower() == "nosniff"
        and headers.get("x-frame-options", "").upper() == "DENY"
    )
    ok("frontend served at / with security headers")
    status, _, nf = call(f"{base}/definitely-not-a-page")
    assert status == 404, status
    ok("unknown page is a 404")

    # 2. what the UI is told about this deployment
    status, _, cfg = jcall(f"{base}/api/v1/config")
    assert status == 200 and cfg["mode"] == "readonly", cfg
    assert cfg["max_question_chars"] and cfg["rate_per_minute"], cfg
    notice = cfg.get("notice") or ""
    ok(
        f"config: mode={cfg['mode']} provider={cfg['llm_provider']} limit={cfg['rate_per_minute']}/min,{cfg['rate_per_day']}/day"
        + (" with the free-tier data notice" if "free tier" in notice else "")
    )

    # 3. privacy and input safeguards
    status, _, fb = jcall(
        f"{base}/api/v1/feedback",
        {"request_id": "x", "rating": "helpful", "answer_status": "answered", "confidence": "high"},
    )
    assert status == 403 and fb["error"]["code"] == "demo_readonly", (status, fb)
    ok("feedback is refused (read-only deployment; nothing stored)")
    status, _, big = jcall(
        f"{base}/api/v1/debug", {"question": "x" * (cfg["max_question_chars"] + 1)}
    )
    assert status == 422 and big["error"]["code"] == "demo_question_too_long", (status, big)
    ok("over-long question rejected before any work")
    status, _, bad = jcall(f"{base}/api/v1/debug", {"question": 5})
    assert status == 422 and "5" not in json.dumps(bad.get("error", {}).get("details", "")), (
        status,
        bad,
    )
    ok("malformed request rejected without echoing the input")
    status, _, trace = jcall(
        f"{base}/api/v1/debug", {"question": OUT_OF_SCOPE, "include_trace": True}
    )
    assert status == 200 and not trace.get("trace"), (
        "the trace must not be exposed in the public deployment"
    )
    assert trace["status"] == "abstained", trace["status"]
    assert trace["timings"]["generation_ms"] == 0, (
        "an out-of-scope question must be declined before the model is called"
    )
    ok(
        f"out-of-scope question declined before the model ({trace['abstain_reason']}); trace not exposed"
    )

    # 4. real answers
    for fw, q in ANSWER_QUESTIONS[: args.answers]:
        started = time.time()
        status, hdrs, body = jcall(f"{base}/api/v1/debug", {"question": q, "framework": fw})
        if status == 429 and body["error"]["code"] == "demo_rate_limited":
            wait = int(hdrs.get("retry-after", "60")) + 1
            print(
                f"...  {fw}: rate limited by design, waiting {wait}s as the Retry-After header says",
                flush=True,
            )
            time.sleep(wait)
            started = time.time()
            status, hdrs, body = jcall(f"{base}/api/v1/debug", {"question": q, "framework": fw})
        if status == 429:
            print(f"SKIP {fw}: {body['error']['code']}, not a failure")
            continue
        assert status == 200, (status, body)
        assert body["status"] in ("answered", "partial"), (
            fw,
            body["status"],
            body.get("abstain_reason"),
        )
        assert body["citations"] and all(c["url"].startswith("https://") for c in body["citations"])
        assert any(c["framework"] == fw for c in body["citations"]), (
            f"{fw} answer should cite {fw} documentation"
        )
        assert body["validation"]["invalid_citations"] == [], body["validation"]
        ok(
            f"{fw}: {body['status']} / {body['confidence']} confidence, {len(body['citations'])} citations, "
            f"model {body['provider']}, {time.time() - started:.1f}s"
        )

    # 5. rate limit (out-of-scope questions are declined by the gate, so no model quota is used)
    if args.check_limits:
        limit = cfg["rate_per_minute"]
        codes = []
        for _ in range(limit + 2):
            status, headers, body = jcall(f"{base}/api/v1/debug", {"question": OUT_OF_SCOPE})
            codes.append(status)
            if status == 429:
                assert headers.get("retry-after", "").isdigit() and body["error"]["code"] in (
                    "demo_rate_limited",
                    "demo_budget_exhausted",
                ), (headers, body)
                break
        assert 429 in codes, f"no 429 after {len(codes)} quick requests: {codes}"
        ok(
            f"rate limit enforced after {codes.index(429)} further requests (Retry-After {headers.get('retry-after')} s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
