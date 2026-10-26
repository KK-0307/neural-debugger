"""Train the from-scratch dual encoder with contrastive (InfoNCE) learning.

    python -m neural_debugger.ml.training.train --config configs/train/small.yaml
    python -m neural_debugger.ml.training.train --config configs/train/small.yaml --seed 1 --device cpu

Everything needed to reproduce a run is written to ``<out_dir>/<name>/seed<k>/``: the config, the
tokenizer, ``best.pt`` (selected checkpoint) and ``last.pt`` (resumable), per-epoch ``metrics.jsonl``
and a ``summary.json`` (data hashes, leakage report, parameters, timings, hardware, seeds).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_debugger.data.corpus import Corpus, load_corpus
from neural_debugger.data.queries import QueryRecord, load_materialised
from neural_debugger.ml.evaluation.qrels import load_qrels
from neural_debugger.ml.retrieval.bm25 import BM25Config, BM25Retriever
from neural_debugger.ml.retrieval.encoders import resolve_device
from neural_debugger.ml.transformer import DualEncoder, TransformerConfig
from neural_debugger.ml.transformer.checkpoint import save_checkpoint

from .config import TrainConfig, load_train_config
from .data import (
    LeakageReport,
    TrainPair,
    filter_leakage,
    generate_synthetic,
    split_by_section,
    summarize,
)
from .evalutils import encode_texts, pool_retrieval_metrics, real_query_metrics
from .losses import info_nce
from .tokenizer import TextTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Prepared:
    tok: TextTokenizer
    train_pairs: list[TrainPair]
    heldout_pairs: list[TrainPair]
    leakage: LeakageReport
    hard_negs: list[list[int]]  # per training pair: candidate negative chunk indices
    summary: dict[str, Any]


def real_pairs(
    corpus: Corpus, queries: list[QueryRecord], relevant: dict[str, set[str]], cap: int = 3
) -> list[TrainPair]:
    by_section: dict[str, list[int]] = {}
    for i, c in enumerate(corpus.chunks):
        by_section.setdefault(c.section_id, []).append(i)
    out: list[TrainPair] = []
    for q in queries:
        for sid in sorted(relevant[q.id]):
            for idx in by_section.get(sid, [])[:cap]:
                out.append(TrainPair(q.query, corpus.chunks[idx].index_text(), idx, sid, "real"))
    return out


def prepare(
    cfg: TrainConfig, corpus: Corpus, queries: list[QueryRecord], tok: TextTokenizer | None = None
) -> Prepared:
    d = cfg.data
    tok = tok or TextTokenizer.train(
        [c.index_text() for c in corpus.chunks], vocab_size=d.vocab_size
    )
    pairs = generate_synthetic(
        corpus.chunks, cfg.seed, d.ict_per_chunk, d.keep_prob, d.heading, d.errors
    )
    # purely synthetic training must not resemble *any* evaluation query; with real supervision the
    # train-split queries are legitimate data, so only val/test are protected.
    protected_splits = {"train", "val", "test"} if d.regime == "synthetic" else {"val", "test"}
    protected = [q.query for q in queries if q.split in protected_splits]
    pairs, leak = filter_leakage(pairs, protected, d.leakage_threshold)
    train_pairs, heldout = split_by_section(pairs, d.holdout_fraction, cfg.seed)
    summary: dict[str, Any] = {
        "synthetic": summarize(train_pairs),
        "heldout": summarize(heldout),
        "leakage": leak.__dict__,
    }
    if d.regime == "synthetic+real":
        tq = [q for q in queries if q.split == "train" and q.slice != "official_faq"]
        rel = load_qrels(tq, corpus, "expanded").relevant
        rp = real_pairs(corpus, tq, rel)
        train_pairs = train_pairs + rp * d.real_repeat
        summary["real"] = {"n_queries": len(tq), "n_pairs": len(rp), "repeat": d.real_repeat}
    hard: list[list[int]] = [[] for _ in train_pairs]
    if cfg.negatives.strategy == "bm25_hard":
        bm25 = BM25Retriever(corpus.chunks, BM25Config())
        cache: dict[str, list[int]] = {}
        for i, p in enumerate(train_pairs):
            if p.query not in cache:
                top = np.argsort(-bm25.score_chunks(p.query), kind="stable")[
                    : cfg.negatives.pool_top + 10
                ]
                cache[p.query] = [int(j) for j in top[cfg.negatives.skip_top :]]
            hard[i] = [j for j in cache[p.query] if corpus.chunks[j].section_id != p.section_id][
                : cfg.negatives.pool_top
            ]
    summary["n_train_pairs_total"] = len(train_pairs)
    return Prepared(tok, train_pairs, heldout, leak, hard, summary)


def drop_tokens(ids: list[int], p: float, rng: random.Random) -> list[int]:
    """Randomly remove non-[CLS] tokens (never empties the sequence)."""
    if p <= 0 or len(ids) <= 2:
        return ids
    kept = [ids[0]] + [t for t in ids[1:] if rng.random() >= p]
    return kept if len(kept) > 1 else ids


def lr_lambda(total: int, warmup: int, min_frac: float):  # type: ignore[no-untyped-def]
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def run_training(
    cfg: TrainConfig,
    corpus: Corpus,
    queries: list[QueryRecord],
    run_dir: Path,
    log: Any = print,
    prepared: Prepared | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    set_seed(cfg.seed)
    device = torch.device(resolve_device(cfg.device))
    if (run_dir / "metrics.jsonl").exists():
        raise FileExistsError(f"{run_dir} already holds a run; use a new directory or --fresh")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=1))
    prep = prepared or prepare(cfg, corpus, queries)
    tok = prep.tok
    tok.save(run_dir / "tokenizer.json")
    mcfg = TransformerConfig(vocab_size=tok.vocab_size, **cfg.model.model_dump())
    model = DualEncoder(mcfg).to(device)
    n_params = model.n_parameters()
    o, dcfg = cfg.optim, cfg.data
    log(
        f"[{cfg.name} seed{cfg.seed}] device={device} params={n_params:,} train_pairs={len(prep.train_pairs):,} "
        f"heldout={len(prep.heldout_pairs):,} leak_dropped={prep.leakage.dropped}"
    )

    # ---- cached tokenisation (ids only; padding happens per batch)
    q_ids = [tok.ids(p.query, dcfg.max_query_len) for p in prep.train_pairs]
    d_ids = [tok.ids(p.doc_text, dcfg.max_doc_len) for p in prep.train_pairs]
    chunk_ids = {
        j: tok.ids(corpus.chunks[j].index_text(), dcfg.max_doc_len)
        for j in {j for h in prep.hard_negs for j in h}
    }
    sec_index = {
        s: i
        for i, s in enumerate(
            sorted({p.section_id for p in prep.train_pairs} | {c.section_id for c in corpus.chunks})
        )
    }

    def pad(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        w = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), w, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
        return ids.to(device), ids.ne(0).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=o.lr, weight_decay=o.weight_decay, betas=(0.9, 0.98)
    )
    steps_per_epoch = len(prep.train_pairs) // o.batch_size
    total = steps_per_epoch * o.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(total, int(o.warmup_frac * total), o.min_lr_frac)
    )
    neg_rng = random.Random(cfg.seed + 7)
    aug_rng = random.Random(cfg.seed + 11)
    aug = cfg.augment

    # ---- validation material
    dev_q = [q for q in queries if q.split in ("train", "val") and q.slice != "official_faq"]
    val_q = [q for q in queries if q.split == "val" and q.slice != "official_faq"]
    qrels = load_qrels(dev_q, corpus, "expanded").relevant if dev_q else {}
    chunk_texts = [c.index_text() for c in corpus.chunks]
    held = prep.heldout_pairs[:2000]
    select_on = "val_ndcg@10" if dcfg.regime == "synthetic+real" else "synth_mrr"

    def evaluate(epoch: int) -> dict[str, Any]:
        model.eval()
        enc_q = lambda i, m: model.encode_query(i, m)  # noqa: E731
        enc_d = lambda i, m: model.encode_doc(i, m)  # noqa: E731
        res: dict[str, Any] = {"epoch": epoch}
        if held:
            hq = encode_texts(enc_q, tok, [p.query for p in held], dcfg.max_query_len, device)
            hd = encode_texts(enc_d, tok, [p.doc_text for p in held], dcfg.max_doc_len, device)
            res.update(
                {
                    f"synth_{k}": v
                    for k, v in pool_retrieval_metrics(
                        hq, hd, [p.section_id for p in held], [p.section_id for p in held]
                    ).items()
                }
            )
        if dev_q:
            d_emb = encode_texts(enc_d, tok, chunk_texts, dcfg.max_doc_len, device)
            q_all = encode_texts(enc_q, tok, [q.query for q in dev_q], dcfg.max_query_len, device)
            m_dev = real_query_metrics(q_all, d_emb, corpus.chunks, dev_q, qrels)
            idx_val = [i for i, q in enumerate(dev_q) if q.split == "val"]
            m_val = real_query_metrics(q_all[idx_val], d_emb, corpus.chunks, val_q, qrels)
            res.update(
                {f"dev_{k}": v for k, v in m_dev.items()}
                | {f"val_{k}": v for k, v in m_val.items()}
            )
        model.train()
        return res

    # ---- loop
    best, best_epoch, bad, history = -1.0, -1, 0, []
    model.train()
    step = 0
    for epoch in range(1, o.epochs + 1):
        perm = np.random.RandomState(cfg.seed * 1000 + epoch).permutation(len(prep.train_pairs))
        losses = []
        t_epoch = time.perf_counter()
        for b in range(steps_per_epoch):
            idx = [int(i) for i in perm[b * o.batch_size : (b + 1) * o.batch_size]]
            qi, qm = pad([drop_tokens(q_ids[i], aug.query_token_dropout, aug_rng) for i in idx])
            di, dm = pad([drop_tokens(d_ids[i], aug.doc_token_dropout, aug_rng) for i in idx])
            groups = torch.tensor(
                [sec_index[prep.train_pairs[i].section_id] for i in idx], device=device
            )
            hard_d = hard_g = None
            if cfg.negatives.strategy == "bm25_hard":
                picks = [
                    neg_rng.sample(
                        prep.hard_negs[i], min(cfg.negatives.n_hard, len(prep.hard_negs[i]))
                    )
                    for i in idx
                    if prep.hard_negs[i]
                ]
                flat = [j for p in picks for j in p]
                if flat:
                    hi, hm = pad([chunk_ids[j] for j in flat])
                    hard_d = model.encode_doc(hi, hm)
                    hard_g = torch.tensor(
                        [sec_index[corpus.chunks[j].section_id] for j in flat], device=device
                    )
            loss = info_nce(
                model.encode_query(qi, qm),
                model.encode_doc(di, dm),
                o.temperature,
                groups,
                hard_d,
                hard_g,
                o.symmetric_loss,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss.detach()))
        ev = evaluate(epoch)
        ev.update(
            {
                "train_loss": float(np.mean(losses)),
                "lr": sched.get_last_lr()[0],
                "epoch_seconds": round(time.perf_counter() - t_epoch, 1),
                "steps": step,
            }
        )
        history.append(ev)
        with (run_dir / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(ev) + "\n")
        score = float(ev.get(select_on, ev.get("synth_mrr", 0.0)))
        log(
            f"  epoch {epoch:2d} loss {ev['train_loss']:.3f} | synth MRR {ev.get('synth_mrr', 0):.3f} | dev NDCG@10 {ev.get('dev_ndcg@10', float('nan')):.3f} "
            f"| val NDCG@10 {ev.get('val_ndcg@10', float('nan')):.3f} | {ev['epoch_seconds']}s"
        )
        meta = {
            "tokenizer_sha256": tok.sha256(),
            "epoch": epoch,
            "config": cfg.model_dump(),
            "select_on": select_on,
        }
        save_checkpoint(run_dir / "last.pt", model, meta)
        if score > best + cfg.stop.min_delta:
            best, best_epoch, bad = score, epoch, 0
            save_checkpoint(run_dir / "best.pt", model, {**meta, "selection_score": score})
        else:
            bad += 1
            if bad >= cfg.stop.patience:
                log(
                    f"  early stopping at epoch {epoch} (best epoch {best_epoch}, {select_on}={best:.4f})"
                )
                break
    summary = {
        "name": cfg.name,
        "seed": cfg.seed,
        "device": str(device),
        "n_parameters": n_params,
        "selection_metric": select_on,
        "best_epoch": best_epoch,
        "best_score": best,
        "epochs_run": len(history),
        "best_epoch_metrics": next(h for h in history if h["epoch"] == best_epoch),
        "data": prep.summary,
        "tokenizer_sha256": tok.sha256(),
        "train_seconds": round(time.perf_counter() - started, 1),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        },
        "config": cfg.model_dump(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--fresh", action="store_true", help="delete an existing run directory first")
    args = ap.parse_args(argv)
    cfg = load_train_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device:
        cfg.device = args.device
    if args.epochs:
        cfg.optim.epochs = args.epochs
    run_dir = (args.out_dir or Path(cfg.out_dir)) / cfg.name / f"seed{cfg.seed}"
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    corpus = load_corpus(args.corpus_dir)
    queries = load_materialised(args.queries)
    # Training code never reads test queries' *labels*; the test split is only used as a leakage shield.
    summary = run_training(cfg, corpus, queries, run_dir)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "name",
                    "seed",
                    "best_epoch",
                    "best_score",
                    "train_seconds",
                    "n_parameters",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
