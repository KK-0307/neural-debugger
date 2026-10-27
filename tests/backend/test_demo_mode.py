"""The public read-only demo mode must not be able to spend quota, store text, or be used as a free general model."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neural_debugger.backend.config import Settings
from neural_debugger.backend.main import create_app
from neural_debugger.backend.services.demo_guard import MAX_TRACKED_CLIENTS, DemoLimiter
from tests.rag.helpers import QUESTION, answer_json, pipeline


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_limiter_enforces_minute_day_and_global_budgets_per_client() -> None:
    clock = Clock()
    lim = DemoLimiter(per_minute=2, per_day=3, global_per_day=4, clock=clock)
    assert lim.check("a").allowed and lim.check("a").allowed
    v = lim.check("a")
    assert not v.allowed and v.code == "demo_rate_limited" and 0 < v.retry_after_s <= 60
    clock.t += 61  # the minute window slides; the daily allowance (3) is still enforced
    assert lim.check("a").allowed
    assert lim.check("a").code == "demo_rate_limited"
    clock.t += 61
    assert lim.check("b").allowed  # another client has its own allowance ...
    v2 = lim.check("c")  # ... until the global daily budget (4 answers across everybody) is used up
    assert not v2.allowed and v2.code == "demo_budget_exhausted" and v2.retry_after_s > 0
    clock.t += 86401
    assert lim.check("c").allowed  # a day later everything resets


def test_limiter_memory_is_bounded_under_a_flood_of_clients() -> None:
    lim = DemoLimiter(per_minute=5, per_day=50, global_per_day=10**9, clock=Clock())
    for i in range(10_500):
        lim.check(f"client-{i}")
    assert len(lim._hits) <= MAX_TRACKED_CLIENTS  # noqa: SLF001
    assert "client-10499" in lim._hits and "client-0" not in lim._hits  # noqa: SLF001  (least recently seen is evicted)


@pytest.fixture
def demo(settings: Settings, tmp_path: Path) -> Iterator[TestClient]:
    cfg = settings.model_copy(
        update={
            "demo_mode": "readonly",
            "demo_max_question_chars": 300,
            "demo_rate_per_minute": 2,
            "demo_rate_per_day": 50,
            "demo_budget_per_day": 3,
            "feedback_dir": str(tmp_path / "fb"),
        }
    )
    with TestClient(create_app(cfg, pipeline(replies=[answer_json()] * 20)[0])) as c:
        yield c


def test_readonly_mode_disables_feedback_and_stores_nothing(demo: TestClient) -> None:
    r = demo.post(
        "/api/v1/feedback",
        json={
            "request_id": "r1",
            "rating": "helpful",
            "question": "private",
            "include_question": True,
        },
    )
    assert r.status_code == 403 and r.json()["error"]["code"] == "demo_readonly"
    assert demo.app.state.feedback.read_all() == []  # type: ignore[attr-defined]


def test_readonly_mode_caps_question_length_and_hides_the_trace(demo: TestClient) -> None:
    long = demo.post("/api/v1/debug", json={"question": "x" * 301})
    assert long.status_code == 422 and long.json()["error"]["code"] == "demo_question_too_long"
    ok = demo.post("/api/v1/debug", json={"question": QUESTION, "include_trace": True})
    assert ok.status_code == 200 and ok.json()["trace"] is None


def test_readonly_mode_rate_limits_with_retry_after_and_a_global_budget(demo: TestClient) -> None:
    assert demo.post("/api/v1/debug", json={"question": QUESTION}).status_code == 200
    assert demo.post("/api/v1/debug", json={"question": QUESTION}).status_code == 200
    third = demo.post("/api/v1/debug", json={"question": QUESTION})
    assert third.status_code == 429 and third.json()["error"]["code"] == "demo_rate_limited"
    assert int(third.headers["retry-after"]) > 0


def test_client_identity_uses_forwarded_for_only_when_the_proxy_is_trusted(
    settings: Settings, tmp_path: Path
) -> None:
    def make(trust: bool) -> TestClient:
        cfg = settings.model_copy(
            update={
                "demo_mode": "readonly",
                "demo_rate_per_minute": 1,
                "demo_rate_per_day": 50,
                "trust_proxy_headers": trust,
                "feedback_dir": str(tmp_path / str(trust)),
            }
        )
        return TestClient(create_app(cfg, pipeline(replies=[answer_json()] * 10)[0]))

    with make(
        False
    ) as spoofable:  # an attacker rotating X-Forwarded-For must NOT get a fresh allowance
        codes = [
            spoofable.post(
                "/api/v1/debug",
                json={"question": QUESTION},
                headers={"X-Forwarded-For": f"1.2.3.{i}"},
            ).status_code
            for i in range(3)
        ]
        assert codes == [200, 429, 429]
    with make(True) as behind_proxy:  # behind a proxy the real client address is the first hop
        codes = [
            behind_proxy.post(
                "/api/v1/debug",
                json={"question": QUESTION},
                headers={"X-Forwarded-For": f"9.9.9.{i}"},
            ).status_code
            for i in range(3)
        ]
        assert codes == [200, 200, 200]


def test_the_default_mode_is_unaffected(settings: Settings, tmp_path: Path) -> None:
    cfg = settings.model_copy(update={"feedback_dir": str(tmp_path / "fb")})
    assert cfg.demo_mode == "off"
    with TestClient(create_app(cfg, pipeline(replies=[answer_json()] * 10)[0])) as c:
        assert all(
            c.post("/api/v1/debug", json={"question": QUESTION}).status_code == 200
            for _ in range(8)
        )
        assert (
            c.post("/api/v1/feedback", json={"request_id": "r", "rating": "helpful"}).status_code
            == 201
        )
