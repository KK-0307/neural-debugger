"""Static guards for the backend Docker image (the image itself is built and smoke-tested in CI)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "docker" / "backend.Dockerfile").read_text()


def _copy_sources() -> list[str]:
    out: list[str] = []
    for line in DOCKERFILE.splitlines():
        m = re.match(r"COPY\s+(?!--from)(.+)", line.strip())
        if m:
            *sources, _dest = m.group(1).split()
            out += sources
    return out


def test_every_copied_path_exists_and_is_not_dockerignored() -> None:
    ignored = [
        ln.strip().rstrip("/")
        for ln in (ROOT / ".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.startswith(("#", "!"))
    ]
    sources = _copy_sources()
    assert {
        "src",
        "data/corpus",
        "data/embeddings",
        "data/corpus_supplement",
        "data/embeddings_rag",
        "pyproject.toml",
        "uv.lock",
    } <= set(sources)
    for src in sources:
        assert (ROOT / src).exists(), f"Dockerfile copies {src} which does not exist"
        assert not any(src == i or src.startswith(i + "/") for i in ignored), (
            f"{src} is dockerignored"
        )


def test_image_installs_the_retrieval_stack_cpu_only_and_stays_offline_without_secrets() -> None:
    assert "--extra neural" in DOCKERFILE
    assert "HF_HUB_OFFLINE=1" in DOCKERFILE
    assert not re.search(r"(?i)(api[_-]?key|secret|password)\s*=", DOCKERFILE)
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "download.pytorch.org/whl/cpu" in pyproject and "sys_platform == 'linux'" in pyproject
    assert "nvidia" not in (ROOT / "uv.lock").read_text(), (
        "CUDA wheels would bloat the image by GBs"
    )


def test_baked_model_revision_comes_from_the_apps_own_config() -> None:
    assert "configs/neural.yaml" in DOCKERFILE and "snapshot_download" in DOCKERFILE
    assert "revision=spec['revision']" in DOCKERFILE


def test_default_settings_only_reference_paths_the_image_contains() -> None:
    from neural_debugger.backend.config import Settings

    s = Settings()
    copied = set(_copy_sources())
    for path in [s.corpus_dir, *s.corpus_supplement_dirs, s.embedding_dir]:
        assert path in copied, (
            f"default setting points at {path}, which the Dockerfile does not copy"
        )
    assert (
        "ND_FEEDBACK_DIR" in DOCKERFILE and "chown" in DOCKERFILE
    )  # the non-root user must be able to write feedback


FULLSTACK = (ROOT / "docker" / "fullstack.Dockerfile").read_text()


def test_fullstack_image_copies_existing_paths_serves_the_exported_site_and_bakes_no_secret() -> (
    None
):
    ignored = [
        ln.strip().rstrip("/")
        for ln in (ROOT / ".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.startswith(("#", "!"))
    ]
    for line in FULLSTACK.splitlines():
        m = re.match(r"COPY\s+(?!--from)(.+)", line.strip())
        if not m:
            continue
        *sources, _dest = m.group(1).split()
        for src in sources:
            assert (ROOT / src).exists(), f"fullstack.Dockerfile copies {src} which does not exist"
            assert not any(src == i or src.startswith(i + "/") for i in ignored), (
                f"{src} is dockerignored"
            )
    assert "NEXT_PUBLIC_EXPORT=1" in FULLSTACK and 'NEXT_PUBLIC_API_URL=""' in FULLSTACK
    assert (
        "ND_STATIC_DIR=/app/site" in FULLSTACK and "COPY --from=site /w/out /app/site" in FULLSTACK
    )
    assert "HF_HUB_OFFLINE=1" in FULLSTACK and "--extra neural" in FULLSTACK
    assert not re.search(r"(?i)(api[_-]?key|secret|password)\s*=", FULLSTACK)


def test_render_blueprint_is_free_plan_only_and_never_contains_the_key() -> None:
    import yaml

    spec = yaml.safe_load((ROOT / "render.yaml").read_text())
    (svc,) = spec["services"]
    assert svc["plan"] == "free" and svc["runtime"] == "docker" and svc["autoDeploy"] is False
    env = {e["key"]: e for e in svc["envVars"]}
    assert env["GEMINI_API_KEY"] == {
        "key": "GEMINI_API_KEY",
        "sync": False,
    }  # the value is entered in the dashboard, not here
    assert (
        env["ND_DEMO_MODE"]["value"] == "readonly" and env["ND_LLM_PROVIDER"]["value"] == "gemini"
    )
    assert int(env["ND_LLM_MAX_REQUESTS"]["value"]) > 0  # a hard cap on model calls
    assert (ROOT / svc["dockerfilePath"]).exists()
