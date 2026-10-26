"""InfoNCE (contrastive) loss with in-batch negatives and optional extra (hard) negatives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def info_nce(
    q: Tensor,
    d: Tensor,
    temperature: float = 0.05,
    group_ids: Tensor | None = None,
    hard_d: Tensor | None = None,
    hard_group_ids: Tensor | None = None,
    symmetric: bool = True,
) -> Tensor:
    """Cross-entropy over similarities; the positive of query ``i`` is document ``i``.

    ``q`` ``(B, E)``, ``d`` ``(B, E)`` (L2-normalised), optional ``hard_d`` ``(M, E)`` extra negatives
    shared by every query. ``group_ids`` ``(B,)`` marks passages that answer the same thing (same
    section): other batch items in the *same group* as query ``i`` are masked out of its negatives, so
    a duplicate positive is never punished. ``symmetric`` adds the document->query direction.
    """
    b = q.size(0)
    cand = d if hard_d is None else torch.cat([d, hard_d], dim=0)  # (B + M, E)
    logits = (q @ cand.T) / temperature  # (B, B + M)
    neg_inf = torch.finfo(logits.dtype).min
    if group_ids is not None:
        cand_groups = (
            group_ids if hard_group_ids is None else torch.cat([group_ids, hard_group_ids])
        )
        same = group_ids[:, None] == cand_groups[None, :]
        same[torch.arange(b), torch.arange(b)] = False  # keep the true positive
        logits = logits.masked_fill(same, neg_inf)
    target = torch.arange(b, device=q.device)
    loss = F.cross_entropy(logits, target)
    if symmetric:
        dq = (d @ q.T) / temperature  # (B, B): each document against all queries
        if group_ids is not None:
            same_qq = group_ids[:, None] == group_ids[None, :]
            same_qq[torch.arange(b), torch.arange(b)] = False
            dq = dq.masked_fill(same_qq, neg_inf)
        loss = 0.5 * (loss + F.cross_entropy(dq, target))
    return loss
