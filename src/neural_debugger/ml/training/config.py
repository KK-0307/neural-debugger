"""Training configuration (YAML -> validated pydantic models)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 512
    max_seq_len: int = 256
    out_dim: int = 128
    dropout: float = 0.1
    pooling: Literal["mean", "cls"] = "mean"
    shared_towers: bool = True


class DataCfg(BaseModel):
    vocab_size: int = 8000
    ict_per_chunk: int = 3
    keep_prob: float = 0.1
    heading: bool = True
    errors: bool = True
    holdout_fraction: float = 0.05  # sections held out for synthetic validation
    leakage_threshold: float = 0.6
    max_query_len: int = 64
    max_doc_len: int = 256
    # "synthetic": documentation-derived pairs only (never sees an evaluation query)
    # "synthetic+real": adds labeled queries from the *train* split (selection then uses val)
    regime: Literal["synthetic", "synthetic+real"] = "synthetic"
    real_repeat: int = 4  # how many times real pairs are repeated per epoch relative to synthetic


class AugCfg(BaseModel):
    query_token_dropout: float = (
        0.0  # randomly drop query tokens (robustness to paraphrase / noise)
    )
    doc_token_dropout: float = 0.0


class NegCfg(BaseModel):
    strategy: Literal["in_batch", "bm25_hard"] = "in_batch"
    n_hard: int = 1  # hard negatives per query
    pool_top: int = 30  # BM25 candidates considered
    skip_top: int = 0  # ignore the very top BM25 hits (likelier false negatives)


class OptimCfg(BaseModel):
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    epochs: int = 12
    batch_size: int = 128
    grad_clip: float = 1.0
    temperature: float = 0.05
    min_lr_frac: float = 0.1
    symmetric_loss: bool = True


class StopCfg(BaseModel):
    patience: int = 3
    min_delta: float = 0.002


class TrainConfig(BaseModel):
    name: str
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs"
    model: ModelCfg = Field(default_factory=ModelCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    negatives: NegCfg = Field(default_factory=NegCfg)
    augment: AugCfg = Field(default_factory=AugCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    stop: StopCfg = Field(default_factory=StopCfg)


def load_train_config(path: Path) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
