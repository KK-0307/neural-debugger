"""Pinned sparse git checkout of a documentation repository (used for FastAPI)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .config import SourceSpec


class GitSourceError(Exception):
    pass


def _git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise GitSourceError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def sparse_patterns(spec: SourceSpec) -> list[str]:
    assert spec.docs_dir is not None
    patterns = [f"/{spec.docs_dir}/{p}" for p in spec.include]
    return [*patterns, "/docs_src/", "/LICENSE"]


def ensure_checkout(spec: SourceSpec, git_dir: Path) -> tuple[Path, str]:
    """Clone (once) at the pinned ref, verify the commit, return (checkout dir, retrieved_at)."""
    assert spec.repo and spec.ref
    meta = git_dir / ".nd_fetch.json"
    if not meta.exists():
        if git_dir.exists():
            raise GitSourceError(f"{git_dir} exists but is not a completed checkout; delete it")
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(
            [
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                spec.ref,
                "--filter=blob:none",
                "--sparse",
                spec.repo,
                str(git_dir),
            ]
        )
        _git(["sparse-checkout", "set", "--no-cone", *sparse_patterns(spec)], cwd=git_dir)
        head = _git(["rev-parse", "HEAD"], cwd=git_dir)
        if spec.commit and head != spec.commit:
            raise GitSourceError(f"{spec.ref} resolved to {head}, config pins {spec.commit}")
        meta.write_text(
            json.dumps(
                {
                    "repo": spec.repo,
                    "ref": spec.ref,
                    "commit": head,
                    "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
    info = json.loads(meta.read_text())
    return git_dir, str(info["retrieved_at"])
