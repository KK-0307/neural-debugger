from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import MultiHeadSelfAttention


class LayerNorm(nn.Module):
    """``(x - mean) / sqrt(var + eps) * gamma + beta`` over the last axis (biased variance)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.up, self.down, self.dropout = (
            nn.Linear(hidden_dim, ffn_dim),
            nn.Linear(ffn_dim, hidden_dim),
            dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.dropout(F.gelu(self.up(x)), self.dropout, self.training))


class EncoderBlock(nn.Module):
    """Pre-LN block: ``x + Attn(LN(x))`` then ``x + FFN(LN(x))`` (residual connections)."""

    def __init__(
        self, hidden_dim: int, n_heads: int, ffn_dim: int, dropout: float, eps: float
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = LayerNorm(hidden_dim, eps), LayerNorm(hidden_dim, eps)
        self.attn = MultiHeadSelfAttention(hidden_dim, n_heads, dropout)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.dropout = dropout

    def forward(self, x: Tensor, keep_mask: Tensor | None = None) -> Tensor:
        x = x + F.dropout(self.attn(self.norm1(x), keep_mask), self.dropout, self.training)
        return x + F.dropout(self.ffn(self.norm2(x)), self.dropout, self.training)
