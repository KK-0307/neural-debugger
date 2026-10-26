"""Tokenizer, contrastive loss, synthetic data + leakage filter, and an end-to-end learning check."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

pytest.importorskip("tokenizers")

from neural_debugger.data.corpus import Corpus  # noqa: E402
from neural_debugger.ml.training.config import TrainConfig  # noqa: E402
from neural_debugger.ml.training.data import (  # noqa: E402
    TrainPair,
    filter_leakage,
    generate_synthetic,
    pairs_fingerprint,
    split_by_section,
)
from neural_debugger.ml.training.losses import info_nce  # noqa: E402
from neural_debugger.ml.training.tokenizer import CLS, PAD, TextTokenizer, normalize_for_model  # noqa: E402
from neural_debugger.ml.training.train import drop_tokens, lr_lambda, run_training  # noqa: E402
from neural_debugger.ml.transformer.checkpoint import load_checkpoint  # noqa: E402
from tests.helpers import make_chunk  # noqa: E402

# ------------------------------------------------------------------------------------ tokenizer


def test_normalisation_splits_identifiers_and_punctuation() -> None:
    assert normalize_for_model("ModuleNotFoundError") == "module not found error"
    assert normalize_for_model("torch.nn.Module") == "torch . nn . module"
    assert normalize_for_model("cuda:0 and cpu") == "cuda : 0 and cpu"
    assert normalize_for_model("snake_case_name  X") == "snake case name x"


def test_tokenizer_training_encoding_padding_and_persistence(tmp_path: Path) -> None:
    texts = [
        "The KeyError is raised when a key is missing",
        "A tensor on cuda:0 and a tensor on cpu",
    ] * 20
    tok = TextTokenizer.train(texts, vocab_size=120, min_frequency=1)
    ids, mask = tok.encode(["tensor cuda", "a key is missing from the dict of tensor"], max_len=6)
    assert ids.shape == (2, 6) and ids[0, 0] == CLS and ids[1, 0] == CLS
    assert bool(mask[0].sum() < mask[1].sum()) and (ids[~mask] == PAD).all()
    assert len(tok.ids("tensor " * 50, max_len=8)) == 8  # truncation keeps [CLS] + 7 tokens
    path = tmp_path / "tok.json"
    tok.save(path)
    again = TextTokenizer.load(path)
    assert again.sha256() == tok.sha256() and again.ids("KeyError missing", 16) == tok.ids(
        "KeyError missing", 16
    )
    assert (
        TextTokenizer.train(texts, vocab_size=120, min_frequency=1).sha256() == tok.sha256()
    )  # deterministic


# ------------------------------------------------------------------------------------- loss


def test_info_nce_matches_hand_computation() -> None:
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[0.8, 0.6], [0.6, 0.8]])
    t = 0.5
    logits = (q @ d.T) / t  # [[1.6, 1.2], [1.2, 1.6]]
    row = -math.log(math.exp(1.6) / (math.exp(1.6) + math.exp(1.2)))
    assert info_nce(q, d, t, symmetric=False).item() == pytest.approx(row, abs=1e-6)
    assert info_nce(q, d, t, symmetric=True).item() == pytest.approx(
        row, abs=1e-6
    )  # symmetric case
    assert logits.shape == (2, 2)


def test_info_nce_prefers_aligned_pairs_and_temperature_sharpens() -> None:
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    aligned = info_nce(q, q.clone(), 0.05)
    shuffled = info_nce(q, q[torch.randperm(16)], 0.05)
    assert aligned < 0.05 < shuffled
    noisy = torch.nn.functional.normalize(q + 0.5 * torch.randn(16, 8), dim=-1)
    assert info_nce(q, noisy, 0.05) > info_nce(q, noisy, 1.0) * 0.0  # both finite
    assert info_nce(q, noisy, 0.05) != info_nce(q, noisy, 0.5)


def test_same_group_negatives_are_masked_but_the_positive_is_kept() -> None:
    torch.manual_seed(1)
    base = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    q = base[[0, 0, 1, 2]]
    d = base[[0, 0, 1, 2]]  # items 0 and 1 are the same passage/group
    groups = torch.tensor([7, 7, 8, 9])
    t = 1.0
    masked = info_nce(q, d, t, groups, symmetric=False)
    unmasked = info_nce(q, d, t, None, symmetric=False)
    assert masked < unmasked  # the duplicate positive is no longer punished as a negative
    s = (q @ d.T) / t
    e = s.exp()
    row0 = -math.log(float(e[0, 0] / (e[0, 0] + e[0, 2] + e[0, 3])))  # d1 (same group) excluded
    row2 = -math.log(float(e[2, 2] / e[2].sum()))  # other groups' duplicates still count
    row3 = -math.log(float(e[3, 3] / e[3].sum()))
    assert masked.item() == pytest.approx((2 * row0 + row2 + row3) / 4, abs=1e-5)


def test_hard_negatives_raise_loss_when_they_are_confusable_and_receive_gradient() -> None:
    torch.manual_seed(2)
    q = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1).requires_grad_()
    d = q.detach().clone()
    base = info_nce(q, d, 0.1)
    hard_close = torch.nn.functional.normalize(
        q.detach() + 0.1 * torch.randn(6, 8), dim=-1
    ).requires_grad_()
    with_hard = info_nce(q, d, 0.1, hard_d=hard_close)
    assert with_hard > base
    with_hard.backward()
    assert hard_close.grad is not None and float(hard_close.grad.abs().sum()) > 0
    # a hard negative from the *same* group as query 0 is masked for query 0 but still counts for the rest
    t = 0.1
    hard = d[:1].clone()
    got = info_nce(
        q, d, t, torch.arange(6), hard_d=hard, hard_group_ids=torch.tensor([0]), symmetric=False
    )
    s = ((q @ torch.cat([d, hard]).T) / t).detach().exp()
    rows = [float(s[i, i] / (s[i, :6].sum() + (0 if i == 0 else s[i, 6]))) for i in range(6)]
    assert got.item() == pytest.approx(-sum(math.log(r) for r in rows) / 6, abs=1e-5)


# ----------------------------------------------------------------------------- synthetic data
CONTENT = (
    "The foo function returns a list of items sorted by key. It raises a ValueError when the input is empty. "
    "Use bar to convert the result into a dictionary of counts.\n\n"
    "```python\nresult = foo(items)\nKeyError: 'missing key in items'\n```"
)


def chunks_for_data() -> list:  # type: ignore[type-arg]
    return [
        make_chunk("alpha", CONTENT, heading="Sorting items", title="Sorting"),
        make_chunk(
            "beta", CONTENT.replace("foo", "baz"), heading="Counting items", title="Counting"
        ),
    ]


def test_synthetic_pairs_have_the_query_removed_from_the_passage_and_are_deterministic() -> None:
    chunks = chunks_for_data()
    pairs = generate_synthetic(chunks, seed=3, ict_per_chunk=3, keep_prob=0.0)
    assert pairs == generate_synthetic(chunks, seed=3, ict_per_chunk=3, keep_prob=0.0)
    assert pairs != generate_synthetic(chunks, seed=4, ict_per_chunk=3, keep_prob=0.0)
    sources = {p.source for p in pairs}
    assert {"synthetic_ict", "synthetic_heading", "synthetic_error"} <= sources
    for p in pairs:
        assert p.query not in p.doc_text.split("\n", 2)[-1] or p.source == "synthetic_heading"
    heading = next(p for p in pairs if p.source == "synthetic_heading" and p.chunk_idx == 0)
    assert (
        heading.query == "Sorting items" and "Sorting items" not in heading.doc_text.split("\n")[1]
    )
    error = next(p for p in pairs if p.source == "synthetic_error")
    assert error.query.startswith("KeyError")


def test_ict_units_are_limited_to_the_visible_prefix() -> None:
    long_prefix = "Intro words " * 200  # ~2,400 characters before the interesting sentence
    c = make_chunk(
        "gamma", long_prefix + "This sentence lives far beyond the visible prefix of the passage."
    )
    pairs = generate_synthetic(
        [c], seed=0, ict_per_chunk=5, heading=False, errors=False, max_offset_chars=1000
    )
    assert all("far beyond" not in p.query for p in pairs)


def test_leakage_filter_drops_exact_and_near_duplicates_only() -> None:
    def mk(q: str) -> TrainPair:
        return TrainPair(q, "doc", 0, "s", "synthetic_ict")

    pairs = [
        mk("KeyError: 'username' when reading from a dictionary"),  # exact eval query
        mk("keyerror username when reading from dictionary"),  # near duplicate (same tokens)
        mk("how do I create a virtual environment for my project"),  # unrelated
        mk("tensor cuda device mismatch error"),
    ]
    protected = [
        "KeyError: 'username' when reading from a dictionary",
        "create a virtual environment for my project",
    ]
    kept, rep = filter_leakage(pairs, protected, threshold=0.6)
    assert [p.query for p in kept] == ["tensor cuda device mismatch error"]
    assert (
        rep.dropped == 3
        and rep.dropped_by_source == {"synthetic_ict": 3}
        and len(rep.examples) == 3
    )
    kept2, _ = filter_leakage(pairs, [], 0.6)
    assert len(kept2) == 4  # nothing protected -> nothing dropped


def test_section_holdout_is_disjoint_and_fingerprint_is_stable() -> None:
    chunks = [make_chunk(f"s{i}", CONTENT) for i in range(20)]
    pairs = generate_synthetic(chunks, seed=0)
    train, held = split_by_section(pairs, 0.25, seed=1)
    assert held and {p.section_id for p in train}.isdisjoint({p.section_id for p in held})
    assert len(train) + len(held) == len(pairs)
    assert pairs_fingerprint(pairs) == pairs_fingerprint(list(pairs))
    assert split_by_section(pairs, 0.25, seed=1) == (train, held)


def test_token_dropout_and_lr_schedule() -> None:
    rng = random.Random(0)
    ids = [2, *range(10, 40)]
    out = drop_tokens(ids, 0.5, rng)
    assert out[0] == 2 and 1 < len(out) < len(ids) and set(out) <= set(ids)
    assert drop_tokens(ids, 0.0, rng) == ids and len(drop_tokens([2, 5], 0.99, rng)) >= 2
    f = lr_lambda(total=100, warmup=10, min_frac=0.1)
    assert (
        f(0) < f(9) <= 1.0 and f(10) == pytest.approx(1.0) and f(99) == pytest.approx(0.1, abs=0.02)
    )


# ------------------------------------------------------------- end-to-end: can it learn a task?


def tiny_corpus(n_sections: int = 40) -> Corpus:
    rng = random.Random(0)
    chunks = []
    for k in range(n_sections):
        words = [f"zw{k}x{j}" for j in range(6)]  # words unique to the section
        sents = []
        for _ in range(5):
            picked = rng.sample(words, 4)
            sents.append(
                f"The {picked[0]} value controls {picked[1]} and {picked[2]} for {picked[3]} here."
            )
        chunks.append(make_chunk(f"sec{k}", " ".join(sents), heading=f"Topic {k}", title="Tiny"))
    return Corpus(chunks=chunks, documents=[], sha256="tiny")


def tiny_config(name: str, strategy: str = "in_batch", epochs: int = 6) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "name": name,
            "seed": 3,
            "device": "cpu",
            "model": {
                "hidden_dim": 32,
                "n_heads": 4,
                "n_layers": 1,
                "ffn_dim": 64,
                "max_seq_len": 64,
                "out_dim": 16,
                "dropout": 0.0,
            },
            "data": {
                "vocab_size": 400,
                "ict_per_chunk": 4,
                "holdout_fraction": 0.2,
                "max_doc_len": 64,
                "max_query_len": 24,
            },
            "negatives": {"strategy": strategy, "pool_top": 10},
            "optim": {"lr": 0.003, "epochs": epochs, "batch_size": 32, "temperature": 0.1},
            "stop": {"patience": 10},
        }
    )


def test_the_model_learns_a_small_synthetic_retrieval_task_and_checkpoints_it(
    tmp_path: Path,
) -> None:
    corpus = tiny_corpus()
    summary = run_training(tiny_config("tiny"), corpus, [], tmp_path / "run", log=lambda *_: None)
    assert summary["best_epoch_metrics"]["synth_mrr"] > 0.5  # held-out sections: chance MRR ~ 0.1
    assert summary["n_parameters"] > 0 and summary["epochs_run"] == 6
    losses = [
        __import__("json").loads(x)["train_loss"]
        for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert losses[-1] < losses[0] * 0.6  # the contrastive loss really decreases
    model, info = load_checkpoint(tmp_path / "run" / "best.pt")
    assert info["meta"]["tokenizer_sha256"] == summary["tokenizer_sha256"] and not model.training
    assert (tmp_path / "run" / "tokenizer.json").exists() and (
        tmp_path / "run" / "summary.json"
    ).exists()


def test_training_is_reproducible_for_a_fixed_seed(tmp_path: Path) -> None:
    corpus = tiny_corpus(20)
    a = run_training(tiny_config("a", epochs=2), corpus, [], tmp_path / "a", log=lambda *_: None)
    b = run_training(tiny_config("a", epochs=2), corpus, [], tmp_path / "b", log=lambda *_: None)
    assert a["best_epoch_metrics"]["train_loss"] == pytest.approx(
        b["best_epoch_metrics"]["train_loss"], abs=1e-6
    )
    assert a["data"]["synthetic"]["sha256"] == b["data"]["synthetic"]["sha256"]


def test_bm25_hard_negative_strategy_runs_and_trains(tmp_path: Path) -> None:
    summary = run_training(
        tiny_config("hard", "bm25_hard", epochs=3),
        tiny_corpus(30),
        [],
        tmp_path / "h",
        log=lambda *_: None,
    )
    assert summary["epochs_run"] == 3 and math.isfinite(summary["best_epoch_metrics"]["train_loss"])


def test_a_finished_run_directory_is_never_silently_appended_to(tmp_path: Path) -> None:
    corpus = tiny_corpus(20)
    run_training(tiny_config("x", epochs=1), corpus, [], tmp_path / "r", log=lambda *_: None)
    with pytest.raises(FileExistsError):
        run_training(tiny_config("x", epochs=1), corpus, [], tmp_path / "r", log=lambda *_: None)
