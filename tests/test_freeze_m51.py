"""The recorded Milestone 5.1 evidence (live runs, adversarial set) must not drift silently."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from neural_debugger.ml.evaluation import freeze

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / freeze.M51_FREEZE_PATH).exists(), reason="no M5.1 freeze file")
def test_m51_evidence_is_intact() -> None:
    assert freeze.verify_m51(ROOT) == []


@pytest.mark.skipif(not (ROOT / freeze.M51_FREEZE_PATH).exists(), reason="no M5.1 freeze file")
def test_m51_drift_is_detected(tmp_path: Path) -> None:
    for rel in [*freeze.M51_FROZEN_FILES, str(freeze.M51_FREEZE_PATH)]:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    assert freeze.verify_m51(tmp_path) == []
    victim = tmp_path / "data/eval/rag/grounding_adversarial.jsonl"
    victim.write_text(victim.read_text() + json.dumps({"tampered": True}) + "\n")
    assert freeze.verify_m51(tmp_path) == ["data/eval/rag/grounding_adversarial.jsonl changed"]
