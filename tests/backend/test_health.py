from fastapi.testclient import TestClient

from neural_debugger.backend import __version__
from neural_debugger.backend.config import Settings
from neural_debugger.backend.main import create_app


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["environment"] == "test"
    assert body["checks"]["database"]["status"] == "ok"
    assert body["checks"]["database"]["latency_ms"] >= 0


def test_health_reports_unreachable_database_with_503() -> None:
    # A real connection failure: the sqlite file's parent directory does not exist.
    bad = Settings(environment="test", database_url="sqlite:////nonexistent-dir/x/nd.db")
    with TestClient(create_app(bad)) as c:
        r = c.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"]["status"] == "error"
    assert body["checks"]["database"]["error_type"] == "OperationalError"


def test_cors_allows_configured_origin(client: TestClient) -> None:
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
