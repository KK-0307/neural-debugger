"""Adapter that exposes the from-scratch transformer dual encoder through the ``Encoder`` protocol.

The checkpoint, the tokenizer and the maximum sequence lengths all determine the embeddings, so the
fingerprint records the SHA-256 of both files and the stored embeddings are rebuilt if either changes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .encoders import Array, resolve_device


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CustomEncoder:
    def __init__(
        self,
        model_dir: str | Path,
        checkpoint: str = "best.pt",
        max_query_len: int = 64,
        max_doc_len: int = 256,
        device: str = "cpu",
        name: str = "custom",
    ) -> None:
        import torch

        from neural_debugger.ml.training.tokenizer import TextTokenizer
        from neural_debugger.ml.transformer.checkpoint import load_checkpoint

        self.name = name
        self.model_dir = Path(model_dir)
        self._ckpt_path = self.model_dir / checkpoint
        self._tok_path = self.model_dir / "tokenizer.json"
        self.resolved_device = resolve_device(device)
        self._torch = torch
        model, info = load_checkpoint(self._ckpt_path)
        self._model = model.to(self.resolved_device).eval()
        self._tok = TextTokenizer.load(self._tok_path)
        if info["meta"].get("tokenizer_sha256") != self._tok.sha256():
            raise ValueError(
                f"{self._tok_path} does not match the tokenizer the checkpoint was trained with"
            )
        self.meta = info["meta"]
        self.max_query_len, self.max_doc_len = max_query_len, max_doc_len
        self.dim = int(self._model.cfg.out_dim)

    def _encode(self, texts: Sequence[str], max_len: int, doc: bool, batch_size: int) -> Array:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        fn = self._model.encode_doc if doc else self._model.encode_query
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with self._torch.no_grad():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                ids, mask = self._tok.encode([texts[i] for i in idx], max_len)
                emb = fn(ids.to(self.resolved_device), mask.to(self.resolved_device))
                out[np.asarray(idx)] = emb.float().cpu().numpy()
        return out

    def encode_documents(self, texts: Sequence[str], batch_size: int = 64) -> Array:
        return self._encode(texts, self.max_doc_len, True, batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> Array:
        return self._encode(texts, self.max_query_len, False, batch_size)

    def fingerprint(self) -> dict[str, Any]:
        return {
            "kind": "custom-transformer",
            "checkpoint_sha256": _sha256(self._ckpt_path),
            "tokenizer_sha256": _sha256(self._tok_path),
            "max_query_len": self.max_query_len,
            "max_doc_len": self.max_doc_len,
            "dim": self.dim,
            "n_parameters": int(self._model.n_parameters()),
            "normalized": True,
        }
