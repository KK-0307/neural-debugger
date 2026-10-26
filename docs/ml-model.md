# Custom transformer dual encoder (Milestone 4)

**Status:** implemented, tested, trained, evaluated once on the held-out test split.
**Headline (honest):** a from-scratch 0.58M-parameter transformer trained on this project's 5k-chunk
documentation corpus **does not beat BM25** and is far behind the pretrained `gte-small` hybrid. Its one
useful role is as a cheap second signal in a hybrid (about 1 ms/query, 2.3 MB), where it roughly matches but
does not exceed BM25 alone. Everything below is measured; every table is generated from files under
`data/eval/results/m4/` (`scripts/render_m4_tables.py`) and `runs/` (`training.aggregate`), not hand-copied.
Labels remain LLM-authored and **not human-verified** (see [evaluation](evaluation.md)).

## 1. What was built

All model code is plain PyTorch tensors and `nn.Linear`/`nn.Embedding`; no `nn.Transformer*`,
`nn.MultiheadAttention` or Hugging Face model classes. Those are used **only as test oracles**.

| Piece | Where | Notes |
|---|---|---|
| Attention | `ml/transformer/attention.py` | explicit scaled dot-product attention, key-padding mask via `finfo.min` (no NaN even for all-pad rows), multi-head split/merge, attention dropout |
| Blocks | `ml/transformer/layers.py` | own `LayerNorm`, GELU feed-forward, pre-LN residual `EncoderBlock` |
| Encoder | `ml/transformer/model.py` | token + learned position embeddings, N blocks, final norm, masked **mean pooling** (or `cls`), linear projection, **L2 normalisation**; `DualEncoder` with shared (default) or separate query/doc towers |
| Checkpoints | `ml/transformer/checkpoint.py` | versioned file, `torch.load(weights_only=True)`, config + metadata + tokenizer hash |
| Tokenizer | `ml/training/tokenizer.py` | BPE (HF `tokenizers`, 8k vocab) after splitting `CamelCase`/`snake_case`/punctuation, so `ModuleNotFoundError` and `torch.nn.Module` become sub-words |
| Loss | `ml/training/losses.py` | InfoNCE with in-batch negatives, temperature, symmetric (q->d and d->q), **same-section false-negative masking**, optional BM25 hard negatives |
| Trainer | `ml/training/train.py` | AdamW (0.9, 0.98), linear warm-up + cosine decay, gradient clipping, seeded, early stopping, best/last checkpoints, per-epoch metrics log |
| Retrieval adapter | `ml/retrieval/custom_encoder.py` | exposes the model through the same `Encoder` protocol as `gte-small`, so dense search, hybrid fusion and all evaluation code are shared |

Correctness checks (`tests/ml/test_transformer.py`, `tests/ml/test_training.py`,
`tests/ml/test_custom_retrieval.py`; 37 tests): attention equals `F.scaled_dot_product_attention` and
`nn.MultiheadAttention` with copied weights; LayerNorm equals `nn.LayerNorm`; outputs have unit norm; padding
does not change an embedding; batch items are independent; position matters; every parameter gets a gradient;
fp64 vs fp32 agree; InfoNCE equals a hand computation, including group masking and hard negatives; the
tokenizer is deterministic; the leakage filter drops exact and near-duplicates only; a tiny model **learns a
synthetic retrieval task** (held-out-section MRR >> chance, loss falls by >40%); runs are reproducible for a
fixed seed; a finished run directory is never appended to.

## 2. Training data (no evaluation query is ever a training target)

There is no large labelled corpus of "debugging question -> documentation section" here, so the model is
trained **only** on pairs generated from the documentation, plus (for one variant) the real *train-split*
queries:

* **inverse cloze (ICT)** a sentence or code line becomes the query and is *removed* from the passage (kept
  10% of the time); only text inside the visible 1,000-character prefix is used because the model truncates
  to 256 tokens (32% of chunks are longer);
* **heading** the section heading is the query, the passage keeps its parents but not that heading;
* **error line** exception lines in code blocks (`KeyError: 'x'`) become queries and are removed.

The shipped regime-A model sees 19,833 pairs over 3,945 sections; 207 sections (1,146 pairs) are held out for
a synthetic validation set that is used for checkpoint selection. A **leakage filter** drops any synthetic
query whose token-set Jaccard with *any* train, val or test evaluation query is >= 0.6 (90 pairs dropped for
the shipped model). Regime B (`tiny-real`) adds the real train-split queries (expanded labels, repeated x3 per
epoch); for B only val and test queries are protected, and the checkpoint is chosen on val NDCG@10.

## 3. Experiments (dev = train + val; 3 seeds each; seed also changes the synthetic split)

| config | seeds | params | best epoch | synth MRR | dev NDCG@10 | dev R@10 | val NDCG@10 | train s |
|---|---|---|---|---|---|---|---|---|
| base | 3 | 5,338,624 | 7.7 | 0.617 ± 0.013 | 0.225 ± 0.017 | 0.306 ± 0.021 | 0.239 ± 0.038 | 905 |
| main | 3 | 1,469,952 | 8.3 | 0.649 ± 0.019 | 0.273 ± 0.018 | 0.340 ± 0.023 | 0.264 ± 0.045 | 191 |
| tiny | 3 | 582,592 | 9.7 | 0.688 ± 0.012 | 0.294 ± 0.012 | 0.366 ± 0.012 | 0.317 ± 0.024 | 86 |
| tiny-hard | 3 | 582,592 | 7.0 | 0.481 ± 0.033 | 0.225 ± 0.009 | 0.266 ± 0.010 | 0.236 ± 0.022 | 194 |
| tiny-real | 3 | 582,592 | 5.3 | 0.653 ± 0.016 | 0.781 ± 0.011 | 0.851 ± 0.016 | 0.369 ± 0.026 | 108 |

`dev NDCG@10` for **tiny-real is in-sample** (it trained on the train queries); only its val figure is a
retrieval result, and even that was used to choose its checkpoint. Synthetic MRR is *not* comparable across
rows with different negative strategies.

What the sweep and the table say:

* **Smaller is better here.** 0.58M parameters beat 1.5M, which beat 5.3M (dev NDCG@10 0.294 vs 0.273 vs
  0.225). With ~20k synthetic pairs and a 5k-chunk corpus the bigger models overfit (train loss keeps falling,
  dev stops improving by epoch 8-10) and cost 2-10x more to train (86 s, 191 s, 905 s on Apple M4 Max/MPS).
* **BM25 hard negatives hurt** (0.225 vs 0.294). Likely cause, not verified: BM25's top results for an ICT query
  are often sibling chunks of the same page that are *also* correct, i.e. false negatives that same-section
  masking cannot catch.
* **Mean pooling >> [CLS] pooling** (single-seed sweep, dev NDCG@10 0.114 vs 0.237); heavier dropout (0.3) hurt;
  light token dropout (query 0.2 / doc 0.1) and more ICT pairs per chunk (8 vs 3) each helped a little. These
  sweep runs were single-seed and on the same dev set used to pick the config, so treat them as directional.
* **Real queries help on val but not on test** (section 5).

## 4. Development results (val = 35 authored queries, held out of both regimes' training)

| system | n | R@1 | R@10 | MRR | NDCG@10 | Δ NDCG@10 vs bm25 | Δ NDCG@10 vs hybrid-gte-tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 35 | 0.284 | 0.653 | 0.702 | 0.574 | – | -0.075 [-0.150, -0.004] |
| hybrid-gte-tuned | 35 | 0.284 | 0.717 | 0.735 | 0.649 | +0.075 [+0.004, +0.150] | – |
| dense-gte-small | 35 | 0.312 | 0.704 | 0.723 | 0.640 | +0.066 [-0.037, +0.172] | -0.009 [-0.062, +0.045] |
| dense-custom-tiny | 35 | 0.122 | 0.419 | 0.421 | 0.345 | -0.230 [-0.357, -0.094] | -0.305 [-0.439, -0.173] |
| hybrid-custom-tiny | 35 | 0.233 | 0.644 | 0.608 | 0.529 | -0.045 [-0.127, +0.034] | -0.120 [-0.230, -0.019] |
| dense-custom-tiny-real | 35 | 0.145 | 0.524 | 0.449 | 0.398 | -0.176 [-0.266, -0.081] | -0.251 [-0.347, -0.162] |
| hybrid-custom-tiny-real | 35 | 0.264 | 0.631 | 0.621 | 0.546 | -0.028 [-0.109, +0.052] | -0.103 [-0.204, -0.014] |

Dev (train + val, 138 authored queries; **the `*-real` rows are in-sample and must not be read as quality**):

| system | n | R@1 | R@10 | MRR | NDCG@10 | Δ NDCG@10 vs bm25 | Δ NDCG@10 vs hybrid-gte-tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 138 | 0.222 | 0.616 | 0.656 | 0.538 | – | -0.125 [-0.165, -0.088] |
| hybrid-gte-tuned | 138 | 0.263 | 0.747 | 0.753 | 0.664 | +0.125 [+0.088, +0.165] | – |
| dense-gte-small | 138 | 0.276 | 0.713 | 0.750 | 0.652 | +0.114 [+0.061, +0.173] | -0.012 [-0.039, +0.017] |
| dense-custom-tiny | 138 | 0.105 | 0.387 | 0.370 | 0.312 | -0.226 [-0.289, -0.165] | -0.352 [-0.412, -0.292] |
| hybrid-custom-tiny | 138 | 0.181 | 0.585 | 0.545 | 0.478 | -0.060 [-0.108, -0.016] | -0.186 [-0.239, -0.135] |
| dense-custom-tiny-real | 138 | 0.339 | 0.862 | 0.822 | 0.791 | +0.253 [+0.184, +0.326] | +0.127 [+0.065, +0.193] |
| hybrid-custom-tiny-real | 138 | 0.300 | 0.797 | 0.781 | 0.711 | +0.173 [+0.129, +0.216] | +0.048 [-0.001, +0.097] |

The official-FAQ slice (n = 23) has a single judged section per query and queries paraphrased from the answer
text, so lexical systems score ~0.98 on it; it says little about the custom model beyond "it is worse":

| system | n | R@1 | R@10 | MRR | NDCG@10 | Δ NDCG@10 vs bm25 | Δ NDCG@10 vs hybrid-gte-tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 23 | 0.957 | 1.000 | 0.978 | 0.984 | – | +0.000 [+0.000, +0.000] |
| hybrid-gte-tuned | 23 | 0.957 | 1.000 | 0.978 | 0.984 | +0.000 [+0.000, +0.000] | – |
| dense-gte-small | 23 | 0.957 | 1.000 | 0.978 | 0.984 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| dense-custom-tiny | 23 | 0.348 | 0.739 | 0.481 | 0.537 | -0.447 [-0.606, -0.287] | -0.447 [-0.606, -0.287] |
| hybrid-custom-tiny | 23 | 0.696 | 1.000 | 0.793 | 0.843 | -0.141 [-0.250, -0.048] | -0.141 [-0.250, -0.048] |
| dense-custom-tiny-real | 23 | 0.652 | 0.957 | 0.751 | 0.799 | -0.185 [-0.304, -0.076] | -0.185 [-0.304, -0.076] |
| hybrid-custom-tiny-real | 23 | 0.739 | 1.000 | 0.870 | 0.904 | -0.080 [-0.144, -0.032] | -0.080 [-0.144, -0.032] |

## 5. Final test (32 authored + 10 FAQ queries; frozen systems, one evaluation per judgment set)

Frozen before the test run: `data/eval/m4_freeze.json` (`make freeze-verify`) hashes the configs, the shipped
weights/tokenizers, the dev results and the M4 judgments. The shipped weights are **seed 0** of each config,
fixed in advance, not the best seed. Test access is logged in `data/eval/results/test_access_log.jsonl`.

**Original labels** (frozen BM25-pooled labels; biased toward BM25):

| system | n | R@1 | R@10 | MRR | NDCG@10 | Δ NDCG@10 vs bm25 | Δ NDCG@10 vs hybrid-gte-tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 32 | 0.329 | 0.692 | 0.666 | 0.597 | – | -0.003 [-0.104, +0.088] |
| hybrid-gte-tuned | 32 | 0.256 | 0.724 | 0.637 | 0.601 | +0.003 [-0.088, +0.104] | – |
| dense-custom-tiny | 32 | 0.121 | 0.370 | 0.363 | 0.306 | -0.292 [-0.403, -0.182] | -0.295 [-0.415, -0.185] |
| hybrid-custom-tiny | 32 | 0.212 | 0.698 | 0.582 | 0.537 | -0.061 [-0.157, +0.028] | -0.064 [-0.181, +0.043] |
| dense-custom-tiny-real | 32 | 0.113 | 0.415 | 0.340 | 0.300 | -0.297 [-0.419, -0.178] | -0.300 [-0.432, -0.179] |
| hybrid-custom-tiny-real | 32 | 0.178 | 0.737 | 0.512 | 0.511 | -0.087 [-0.168, -0.010] | -0.090 [-0.199, +0.023] |

**Expanded labels** (original + M4 pool review of the systems' top-5; the *test* part of that review was done
**after seeing the test rankings**, so it is post-hoc, disclosed here and kept in a separate file,
`pooled_judgments_m4_test.yaml`; it added 3 relevant sections over 32 queries):

| system | n | R@1 | R@10 | MRR | NDCG@10 | Δ NDCG@10 vs bm25 | Δ NDCG@10 vs hybrid-gte-tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 32 | 0.309 | 0.630 | 0.666 | 0.560 | – | -0.061 [-0.164, +0.036] |
| hybrid-gte-tuned | 32 | 0.244 | 0.748 | 0.671 | 0.621 | +0.061 [-0.036, +0.164] | – |
| dense-custom-tiny | 32 | 0.117 | 0.377 | 0.382 | 0.314 | -0.246 [-0.354, -0.133] | -0.307 [-0.424, -0.193] |
| hybrid-custom-tiny | 32 | 0.197 | 0.672 | 0.585 | 0.531 | -0.029 [-0.131, +0.062] | -0.090 [-0.205, +0.018] |
| dense-custom-tiny-real | 32 | 0.106 | 0.419 | 0.361 | 0.309 | -0.251 [-0.372, -0.131] | -0.312 [-0.441, -0.194] |
| hybrid-custom-tiny-real | 32 | 0.174 | 0.707 | 0.516 | 0.507 | -0.052 [-0.132, +0.022] | -0.113 [-0.224, -0.005] |

FAQ slice (original labels):

| system | n | R@1 | R@10 | MRR | NDCG@10 | Δ NDCG@10 vs bm25 | Δ NDCG@10 vs hybrid-gte-tuned |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 10 | 0.900 | 1.000 | 0.950 | 0.963 | – | -0.037 [-0.111, +0.000] |
| hybrid-gte-tuned | 10 | 1.000 | 1.000 | 1.000 | 1.000 | +0.037 [+0.000, +0.111] | – |
| dense-custom-tiny | 10 | 0.500 | 0.800 | 0.624 | 0.662 | -0.301 [-0.591, -0.027] | -0.338 [-0.600, -0.101] |
| hybrid-custom-tiny | 10 | 0.900 | 0.900 | 0.909 | 0.900 | -0.063 [-0.300, +0.111] | -0.100 [-0.300, +0.000] |
| dense-custom-tiny-real | 10 | 0.500 | 0.800 | 0.637 | 0.676 | -0.287 [-0.574, -0.026] | -0.324 [-0.587, -0.100] |
| hybrid-custom-tiny-real | 10 | 0.700 | 0.900 | 0.784 | 0.806 | -0.157 [-0.400, +0.000] | -0.194 [-0.425, +0.000] |

Reading the test results:

* The custom **dense** model alone is ~0.25-0.30 NDCG@10 below BM25 with a paired interval that excludes zero
  under both label sets. The M3 result stands: pretrained dense/hybrid retrieval is the better system.
* The custom **hybrid** with BM25 is below BM25 on NDCG@10 (-0.03 to -0.09) but with intervals that include
  zero for regime A; its Recall@10 is about equal to BM25's (0.70 vs 0.69 original labels). With 32 queries
  this is "not distinguishable from BM25", **not** "as good as BM25".
* Regime B's val advantage (0.398 vs 0.345 dense NDCG@10) **did not carry over to test** (0.30 vs 0.31).
  Selecting on a 35-query val set and tuning on the same set is noisy; I do not claim real supervision does or
  does not help at this scale.
* Unjudged@10 was 0.5-0.75 for the custom systems under the original labels and 0.19-0.33 after M4 pooling;
  condensed metrics are in each system's JSON. Judgments are by the same assistant that built the system.

## 6. Cost (Apple M4 Max, CPU inference, 4,995 chunks)

| system | params | model file | build / load | embed corpus (one-off) | p50 / p95 query (ms) | RSS after build |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | – | – | 0.7 s | – | 0.71 / 0.86 | 159 MB |
| dense-custom-tiny | 582,592 | 2.3 MB (+0.5 MB tokenizer) | 1.1 s | 1.4 s (CPU) | 1.05 / 1.34 | 309 MB (mostly PyTorch import) |
| hybrid-custom-tiny | same | same | 1.7 s | – | 2.06 / 2.37 | 326 MB |
| hybrid-gte-tuned (M3) | 33M | 67 MB | 6.0 s | 10.7 s (MPS) | 7.99 / 9.74 | 528 MB |

The custom model is ~8x faster per query and ~30x smaller than `gte-small`, which is its only advantage.

## 7. Limitations and what is not claimed

* **Not better than the baselines**, and the evaluation is small (35 val / 32 test authored queries): most
  paired intervals are wide. Dev numbers carry selection optimism (best of 3 sizes, a sweep, 2 regimes).
* **Labels are LLM-authored and unreviewed by a human**; all pool expansion (M3 and M4) was done by the same
  assistant, and the test pool expansion was post-hoc.
* **Data-starved by construction:** ~20k synthetic pairs from a 5k-chunk corpus is orders of magnitude below
  what dense retrievers are normally trained on; no pretrained initialisation or distillation was used (that
  would make it a fine-tuning project, not a from-scratch one). Larger synthetic data, LLM-generated queries
  or mining real Stack Overflow pairs are untried and would need approval (licences).
* **No mid-run resume**: checkpoints (`best.pt`, `last.pt`) hold model weights and metadata for selection and
  inference, not optimiser/RNG state. Training is cheap (1.5-15 min) and reproducible from the seed.
* One tokenizer vocabulary size (8k), one temperature (0.05), no learning-rate sweep, no multi-vector or
  cross-attention model, no attention-visualisation tooling (attention weights are returned by the module but
  unused).
* Training uses MPS; CPU results differ in the last bits. Embeddings are rebuilt on CPU for evaluation.
* Seeds change both initialisation and the synthetic held-out split, so seed variance includes data variance.

## 8. Reproduce

```bash
uv sync --extra ml --extra neural --extra data --extra index
export PYTHONPATH=$PWD/src     # macOS hidden-.pth workaround, see neural-retrieval.md
python -m neural_debugger.ml.training.train --config configs/train/tiny.yaml --seed 0 --out-dir runs/m4
python -m neural_debugger.ml.training.aggregate runs/m4
python -m neural_debugger.ml.evaluation.system_eval --system hybrid-custom-tiny \
    --extra-config configs/m4.yaml --qrels m4 --splits train val          # dev only
make freeze-verify                                                         # M2 + M3 + M4 manifests
```

Shipped weights (`models/custom-tiny`, `models/custom-tiny-real`) are tracked despite `*.pt` being ignored.
The test split needs `--final-test` and is logged.
