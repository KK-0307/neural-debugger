from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformerConfig:
    """Hyper-parameters of the encoder. Defaults are the CPU-friendly sizes from the project brief."""

    vocab_size: int = 8000
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 512
    max_seq_len: int = 256
    out_dim: int = 128
    dropout: float = 0.1
    pooling: str = "mean"  # "mean" | "cls"
    pad_id: int = 0
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    shared_towers: bool = True  # one encoder for queries and documents (siamese) or two
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hidden_dim % self.n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransformerConfig:
        return cls(**d)
