"""Model checkpoints: weights + the exact config + provenance, loadable without pickle code execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import TransformerConfig
from .model import DualEncoder

FORMAT_VERSION = 1


def save_checkpoint(
    path: Path,
    model: DualEncoder,
    meta: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "config": model.cfg.as_dict(),
            "model_state": model.state_dict(),
            "meta": meta or {},
            "extra_state": extra_state or {},
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[DualEncoder, dict[str, Any]]:
    """Return ``(model in eval mode, meta)``. Uses ``weights_only=True`` (tensors + primitives only)."""
    blob = torch.load(path, map_location=map_location, weights_only=True)
    if blob.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {blob.get('format_version')!r}")
    model = DualEncoder(TransformerConfig.from_dict(blob["config"]))
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, {"meta": blob["meta"], "extra_state": blob["extra_state"]}
