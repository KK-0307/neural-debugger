"""One container serves the exported frontend and the API; the UI learns the deployment mode from /api/v1/config."""

from pathlib import Path

from fastapi.testclient import TestClient

from neural_debugger.backend.config import Settings
from neural_debugger.backend.main import create_app


def _site(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<!doctype html><title>Neural Debugger</title>home")
    (site / "404.html").write_text("<!doctype html>missing page")
    return site


def test_the_api_alone_has_no_site_and_reports_mode_off(settings: Settings) -> None:
    with TestClient(create_app(settings)) as c:
        assert c.get("/").status_code == 404
        body = c.get("/api/v1/config").json()
    assert body["mode"] == "off" and body["max_question_chars"] is None and body["notice"] is None


def test_config_reports_limits_and_the_free_tier_notice_only_when_they_apply(
    settings: Settings,
) -> None:
    cfg = settings.model_copy(
        update={"demo_mode": "readonly", "llm_provider": "gemini", "demo_max_question_chars": 1234}
    )
    with TestClient(create_app(cfg)) as c:
        body = c.get("/api/v1/config").json()
    assert body["mode"] == "readonly" and body["max_question_chars"] == 1234
    assert body["rate_per_minute"] == cfg.demo_rate_per_minute
    assert "free tier" in body["notice"] and "improve" in body["notice"]
    assert "key" not in " ".join(
        k for k in body if k != "notice"
    )  # nothing secret-shaped is exposed


def test_the_exported_site_is_served_at_the_root_without_shadowing_the_api(
    settings: Settings, tmp_path: Path
) -> None:
    cfg = settings.model_copy(update={"static_dir": str(_site(tmp_path))})
    with TestClient(create_app(cfg)) as c:
        home = c.get("/")
        assert home.status_code == 200 and "Neural Debugger" in home.text
        assert c.get("/health").json()["version"]  # API routes win over the static mount
        assert c.get("/api/v1/config").status_code == 200
        missing = c.get("/no-such-page")
        assert missing.status_code == 404 and "missing page" in missing.text
        assert (
            c.post("/api/v1/debug", json={"question": "x"}).status_code == 422
        )  # still the API's validation


def test_security_headers_are_set_on_api_and_site(settings: Settings, tmp_path: Path) -> None:
    cfg = settings.model_copy(update={"static_dir": str(_site(tmp_path))})
    with TestClient(create_app(cfg)) as c:
        for path in ("/", "/health"):
            r = c.get(path)
            assert r.headers["x-content-type-options"] == "nosniff"
            assert r.headers["x-frame-options"] == "DENY"
            assert r.headers["referrer-policy"]
