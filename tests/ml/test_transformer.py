"""Correctness tests for the from-scratch transformer (shapes, masking, padding, gradients,
numerical stability, determinism, checkpointing) and comparison against trusted PyTorch references."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from neural_debugger.ml.transformer import DualEncoder, TransformerConfig, TransformerEncoder  # noqa: E402
from neural_debugger.ml.transformer.attention import (  # noqa: E402
    MultiHeadSelfAttention,
    scaled_dot_product_attention,
)
from neural_debugger.ml.transformer.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from neural_debugger.ml.transformer.layers import LayerNorm  # noqa: E402
from neural_debugger.ml.transformer.model import SequenceEncoder, masked_mean_pool  # noqa: E402

CFG = TransformerConfig(
    vocab_size=50,
    hidden_dim=32,
    n_heads=4,
    n_layers=2,
    ffn_dim=64,
    max_seq_len=16,
    out_dim=16,
    dropout=0.0,
)


def batch(
    lengths: list[int], seed: int = 0, pad_to: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    s = pad_to or max(lengths)
    ids = torch.zeros(len(lengths), s, dtype=torch.long)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(1, CFG.vocab_size, (n,), generator=g)
    return ids, ids.ne(0)


# ------------------------------------------------------------------------- references
def test_scaled_dot_product_attention_matches_pytorch_reference() -> None:
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 4, 7, 8, generator=g) for _ in range(3))
    ours, w = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 4, 7))  # each query's weights sum to 1


def test_masked_attention_matches_reference_with_boolean_mask() -> None:
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(3, 2, 6, 8, generator=g) for _ in range(3))
    keep = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    ours, w = scaled_dot_product_attention(q, k, v, keep)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)
    assert torch.all(
        w.masked_select(~keep[:, None, None, :].expand_as(w)) == 0
    )  # exactly zero on pads


def test_multi_head_attention_matches_torch_nn_multihead_attention() -> None:
    d, h = 32, 4
    ours = MultiHeadSelfAttention(d, h).eval()
    ref = nn.MultiheadAttention(d, h, batch_first=True).eval()
    with torch.no_grad():  # copy our q/k/v/out projections into torch's packed layout
        ref.in_proj_weight.copy_(
            torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight])
        )
        ref.in_proj_bias.copy_(torch.cat([ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias]))
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        ref.out_proj.bias.copy_(ours.out_proj.bias)
    x = torch.randn(3, 9, d, generator=torch.Generator().manual_seed(3))
    keep = torch.tensor([[1] * 9, [1] * 5 + [0] * 4, [1] * 2 + [0] * 7], dtype=torch.bool)
    out = ours(x, keep)
    ref_out, _ = ref(x, x, x, key_padding_mask=~keep, need_weights=False)
    valid = keep.unsqueeze(-1).expand_as(out)  # outputs at padded query positions are unspecified
    torch.testing.assert_close(out[valid], ref_out[valid], atol=1e-5, rtol=1e-5)


def test_layer_norm_matches_torch_layer_norm() -> None:
    x = torch.randn(4, 5, 32, generator=torch.Generator().manual_seed(4)) * 5 + 3
    ours, ref = LayerNorm(32), nn.LayerNorm(32, eps=1e-5)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(32)), ours.bias.copy_(torch.randn(32))
        ref.weight.copy_(ours.weight), ref.bias.copy_(ours.bias)
    torch.testing.assert_close(ours(x), ref(x), atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------- shapes / masking / padding
def test_output_shapes_and_unit_norm() -> None:
    ids, mask = batch([5, 9, 3])
    enc = TransformerEncoder(CFG).eval()
    assert enc(ids, mask).shape == (3, 9, CFG.hidden_dim)
    emb = SequenceEncoder(CFG).eval()(ids, mask)
    assert emb.shape == (3, CFG.out_dim)
    torch.testing.assert_close(emb.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=0)


def test_padding_never_changes_real_token_states_or_the_embedding() -> None:
    torch.manual_seed(0)
    enc = SequenceEncoder(CFG).eval()
    short, _ = batch([5], seed=7)
    padded = torch.cat([short, torch.zeros(1, 8, dtype=torch.long)], dim=1)  # same tokens + 8 pads
    torch.testing.assert_close(enc(short), enc(padded), atol=1e-5, rtol=1e-5)
    h_short = enc.encoder(short)
    h_pad = enc.encoder(padded)
    torch.testing.assert_close(h_short, h_pad[:, :5], atol=1e-5, rtol=1e-5)


def test_changing_a_pad_position_token_embedding_does_not_leak_into_outputs() -> None:
    torch.manual_seed(0)
    enc = SequenceEncoder(CFG).eval()
    ids, mask = batch([4, 7], seed=5)
    base = enc(ids, mask)
    ids2 = ids.clone()
    ids2[0, 6] = 9  # a masked-out position of the first sequence now holds a real-looking token id
    torch.testing.assert_close(
        base[0], enc(ids2, mask)[0], atol=1e-5, rtol=1e-5
    )  # explicit mask => ignored


def test_batch_independence() -> None:
    torch.manual_seed(0)
    enc = SequenceEncoder(CFG).eval()
    ids, mask = batch([6, 3, 8], seed=9)
    together = enc(ids, mask)
    for i in range(3):
        torch.testing.assert_close(
            together[i : i + 1], enc(ids[i : i + 1], mask[i : i + 1]), atol=1e-5, rtol=1e-5
        )


def test_model_is_position_aware_but_mean_pool_ignores_pad_count() -> None:
    torch.manual_seed(0)
    enc = SequenceEncoder(CFG).eval()
    ids, _ = batch([8], seed=11)
    assert not torch.allclose(enc(ids), enc(ids.flip(1)), atol=1e-4)  # order matters via positions
    hidden = torch.randn(2, 5, 4)
    keep = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    pooled = masked_mean_pool(hidden, keep)
    torch.testing.assert_close(pooled[0], hidden[0, :2].mean(0))
    assert torch.all(pooled[1] == 0)  # empty row -> zeros, not NaN


def test_sequence_longer_than_max_len_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        TransformerEncoder(CFG)(torch.ones(1, CFG.max_seq_len + 1, dtype=torch.long))


# -------------------------------------------------------------------- gradients / stability
def test_gradients_reach_every_parameter() -> None:
    torch.manual_seed(0)
    model = DualEncoder(CFG)
    q, qm = batch([5, 6], seed=1)
    d, dm = batch([9, 12], seed=2)
    loss = (model.encode_query(q, qm) * model.encode_doc(d, dm)).sum()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    dead = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) == 0.0
    ]
    assert not missing, f"parameters without a gradient: {missing}"
    # nothing may be dead except (at most) the token-embedding table's own bookkeeping
    assert all("token_emb" in n for n in dead), f"parameters with identically-zero gradient: {dead}"
    total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert total > 0


def test_numerical_stability_with_extreme_inputs() -> None:
    torch.manual_seed(0)
    enc = SequenceEncoder(CFG).eval()
    ids, mask = batch([4, 16, 1], seed=3)
    with torch.no_grad():
        enc.encoder.token_emb.weight.mul_(1e3)  # very large activations
        out = enc(ids, mask)
    assert torch.isfinite(out).all()
    all_pad = torch.zeros(2, 5, dtype=torch.long)  # sequences with no real token at all
    out_pad = enc(all_pad, torch.zeros(2, 5, dtype=torch.bool))
    assert torch.isfinite(out_pad).all()  # no NaN from fully-masked softmax rows


def test_double_precision_agrees_with_single_precision() -> None:
    torch.manual_seed(0)
    enc = SequenceEncoder(CFG).eval()
    ids, mask = batch([7, 4], seed=4)
    a = enc(ids, mask)
    b = SequenceEncoder(CFG).eval().double()
    b.load_state_dict({k: v.double() for k, v in enc.state_dict().items()})
    torch.testing.assert_close(a.double(), b(ids, mask), atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------ determinism / dropout / checkpoint
def test_inference_is_deterministic_and_seeded_init_is_reproducible() -> None:
    ids, mask = batch([6, 9], seed=5)
    torch.manual_seed(42)
    m1 = DualEncoder(CFG).eval()
    torch.manual_seed(42)
    m2 = DualEncoder(CFG).eval()
    assert all(
        torch.equal(a, b)
        for a, b in zip(m1.state_dict().values(), m2.state_dict().values(), strict=True)
    )
    assert torch.equal(m1.encode_doc(ids, mask), m1.encode_doc(ids, mask))
    assert torch.equal(m1.encode_doc(ids, mask), m2.encode_doc(ids, mask))


def test_dropout_only_active_in_training_mode() -> None:
    cfg = TransformerConfig(**{**CFG.as_dict(), "dropout": 0.5})
    model = SequenceEncoder(cfg)
    ids, mask = batch([8, 8], seed=6)
    model.train()
    assert not torch.equal(model(ids, mask), model(ids, mask))
    model.eval()
    assert torch.equal(model(ids, mask), model(ids, mask))


def test_checkpoint_round_trip_preserves_outputs_and_config(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = DualEncoder(CFG).eval()
    ids, mask = batch([5, 8], seed=8)
    path = tmp_path / "ckpt" / "m.pt"
    save_checkpoint(path, model, meta={"tokenizer_sha256": "abc", "step": 7})
    loaded, info = load_checkpoint(path)
    assert loaded.cfg == model.cfg and info["meta"] == {"tokenizer_sha256": "abc", "step": 7}
    assert not loaded.training
    assert torch.equal(model.encode_query(ids, mask), loaded.encode_query(ids, mask))
    bad = tmp_path / "bad.pt"
    torch.save({"format_version": 99}, bad)
    with pytest.raises(ValueError, match="unsupported"):
        load_checkpoint(bad)


def test_shared_and_separate_towers() -> None:
    shared = DualEncoder(CFG)
    sep = DualEncoder(TransformerConfig(**{**CFG.as_dict(), "shared_towers": False}))
    assert shared.query_tower is shared.doc_tower and sep.query_tower is not sep.doc_tower
    assert sep.n_parameters() > shared.n_parameters()
    ids, mask = batch([5], seed=1)
    assert not torch.equal(sep.encode_query(ids, mask), sep.encode_doc(ids, mask))


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        TransformerConfig(hidden_dim=30, n_heads=4)
    with pytest.raises(ValueError):
        TransformerConfig(pooling="max")
    assert TransformerConfig.from_dict(CFG.as_dict()) == CFG
