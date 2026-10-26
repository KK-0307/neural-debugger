"""Scaled dot-product and multi-head self-attention, with explicit tensor shapes.

Shapes (B = batch, S = sequence length, H = heads, Dh = head dim, D = H * Dh):

    x            (B, S, D)
    q, k, v      (B, H, S, Dh)        after projection and head split
    scores       (B, H, S, S)         q @ k^T / sqrt(Dh)
    keep_mask    (B, S) bool          True for real tokens, False for padding
    weights      (B, H, S, S)         softmax over the key axis
    context      (B, H, S, Dh)  ->  (B, S, D) after merging heads
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    keep_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """``softmax(q k^T / sqrt(Dh)) v`` with padding keys excluded.

    Masked scores are set to the most negative finite value (not ``-inf``) so a fully padded row
    cannot produce NaN; such rows get all-zero attention weights instead.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    weights_mask = None
    if keep_mask is not None:
        weights_mask = keep_mask[:, None, None, :]  # (B, 1, 1, S): broadcast over heads and queries
        scores = scores.masked_fill(~weights_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if weights_mask is not None:
        # exact zeros on padding keys, and zero rows when a sequence has no real token at all
        weights = weights * weights_mask.to(weights.dtype)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return torch.matmul(weights, v), weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads, self.head_dim, self.dropout = n_heads, hidden_dim // n_heads, dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _split(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, Dh)

    def forward(self, x: Tensor, keep_mask: Tensor | None = None, return_weights: bool = False):  # type: ignore[no-untyped-def]
        q, k, v = (
            self._split(self.q_proj(x)),
            self._split(self.k_proj(x)),
            self._split(self.v_proj(x)),
        )
        context, weights = scaled_dot_product_attention(
            q, k, v, keep_mask, self.dropout, self.training
        )
        b, _, s, _ = context.shape
        merged = context.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.head_dim)
        out = self.out_proj(merged)
        return (out, weights) if return_weights else out
