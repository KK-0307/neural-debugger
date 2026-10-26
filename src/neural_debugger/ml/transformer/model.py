"""Transformer encoder, sequence encoder (pooling + projection + L2 norm) and dual encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import TransformerConfig
from .layers import EncoderBlock, LayerNorm


class TransformerEncoder(nn.Module):
    """token ids ``(B, S)`` -> contextual states ``(B, S, D)``."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.hidden_dim, cfg.n_heads, cfg.ffn_dim, cfg.dropout, cfg.layer_norm_eps)
            for _ in range(cfg.n_layers)
        )
        self.final_norm = LayerNorm(cfg.hidden_dim, cfg.layer_norm_eps)
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.cfg.init_std)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.cfg.init_std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        b, s = input_ids.shape
        if s > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {s} exceeds max_seq_len {self.cfg.max_seq_len}")
        keep = (
            attention_mask.bool() if attention_mask is not None else input_ids.ne(self.cfg.pad_id)
        )
        positions = torch.arange(s, device=input_ids.device)[None, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = F.dropout(x, self.cfg.dropout, self.training)
        for block in self.blocks:
            x = block(x, keep)
        return self.final_norm(x)


def masked_mean_pool(hidden: Tensor, keep_mask: Tensor) -> Tensor:
    """Average of the real-token states: padding contributes nothing, empty rows give zeros."""
    m = keep_mask.to(hidden.dtype).unsqueeze(-1)  # (B, S, 1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class SequenceEncoder(nn.Module):
    """ids -> L2-normalised embedding ``(B, out_dim)`` (so dot product == cosine similarity)."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(cfg)
        self.projection = nn.Linear(cfg.hidden_dim, cfg.out_dim, bias=False)
        nn.init.normal_(self.projection.weight, std=cfg.init_std)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        keep = (
            attention_mask.bool() if attention_mask is not None else input_ids.ne(self.cfg.pad_id)
        )
        hidden = self.encoder(input_ids, keep)
        pooled = hidden[:, 0] if self.cfg.pooling == "cls" else masked_mean_pool(hidden, keep)
        return F.normalize(self.projection(pooled), p=2.0, dim=-1, eps=1e-12)


class DualEncoder(nn.Module):
    """Query and document towers (shared weights by default)."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.query_tower = SequenceEncoder(cfg)
        self.doc_tower = self.query_tower if cfg.shared_towers else SequenceEncoder(cfg)

    def encode_query(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.query_tower(input_ids, attention_mask)

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.doc_tower(input_ids, attention_mask)

    def n_parameters(self) -> int:
        return sum(
            p.numel() for p in self.parameters()
        )  # shared tower counted once by .parameters()
