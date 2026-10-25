import pytest

from neural_debugger.backend.config import Settings


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ND_DATABASE_URL", "sqlite:///x.db")
    monkeypatch.setenv("ND_ENVIRONMENT", "production")
    monkeypatch.setenv("ND_CORS_ORIGINS", '["https://app.example"]')
    s = Settings()
    assert s.database_url == "sqlite:///x.db"
    assert s.environment == "production"
    assert s.cors_origins == ["https://app.example"]


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ND_ENVIRONMENT", "staging-ish")
    with pytest.raises(ValueError):
        Settings()
