from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neural_debugger.backend.config import Settings
from neural_debugger.backend.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(environment="test", database_url=f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c
