"""Encoding and retrieval metrics for the custom model (used during training and at evaluation time)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from neural_debugger.data.queries import QueryRecord
from neural_debugger.data.schema import Chunk
from neural_debugger.ml.evaluation.metrics import query_metrics

from .tokenizer import TextTokenizer


@torch.no_grad()
def encode_texts(
    encode: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    tok: TextTokenizer,
    texts: Sequence[str],
    max_len: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Embed ``texts`` (length-sorted batches, original order restored) -> float32 ``(N, E)``."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: np.ndarray | None = None
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        ids, mask = tok.encode([texts[i] for i in idx], max_len)
        emb = encode(ids.to(device), mask.to(device)).float().cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
        out[np.asarray(idx)] = emb
    return out if out is not None else np.zeros((0, 0), dtype=np.float32)


def rank_sections(scores: np.ndarray, chunks: Sequence[Chunk], depth: int = 100) -> list[str]:
    """Best-chunk-per-section ranking from one query's chunk scores (ties -> lower index)."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    seen: set[str] = set()
    ranking: list[str] = []
    for i in order:
        sid = chunks[int(i)].section_id
        if sid not in seen:
            seen.add(sid)
            ranking.append(sid)
            if len(ranking) >= depth:
                break
    return ranking


def real_query_metrics(
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    chunks: Sequence[Chunk],
    queries: Sequence[QueryRecord],
    relevant: dict[str, set[str]],
) -> dict[str, float]:
    per = [
        query_metrics(rank_sections(d_emb @ q_emb[i], chunks), relevant[q.id])
        for i, q in enumerate(queries)
    ]
    if not per:
        return {}
    return {
        k: float(np.mean([m[k] for m in per])) for k in ("ndcg@10", "mrr", "recall@10", "recall@1")
    }


def pool_retrieval_metrics(
    q_emb: np.ndarray, d_emb: np.ndarray, d_groups: Sequence[str], q_groups: Sequence[str]
) -> dict[str, float]:
    """Retrieve among a pool of documents; a hit is any document of the query's group (section)."""
    scores = q_emb @ d_emb.T
    rr, r1, r10 = [], [], []
    groups = np.asarray(list(d_groups))
    for i in range(len(q_emb)):
        order = np.argsort(-scores[i], kind="stable")
        pos = np.flatnonzero(groups[order] == q_groups[i])
        rank = int(pos[0]) + 1 if len(pos) else len(order) + 1
        rr.append(1.0 / rank)
        r1.append(float(rank <= 1))
        r10.append(float(rank <= 10))
    return {
        "mrr": float(np.mean(rr)),
        "recall@1": float(np.mean(r1)),
        "recall@10": float(np.mean(r10)),
        "pool": float(len(d_emb)),
    }
