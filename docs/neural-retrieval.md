# Neural retrieval (Milestone 3): dense, hybrid and reranked search

**Status:** implemented, tested, and evaluated. All numbers below come from files under
`data/eval/results/m3/` (regenerate the tables with `scripts/render_m3_tables.py`). Read the
[limitations](#9-limitations-and-what-is-not-claimed) before quoting any figure: the test split has
32 authored queries, the labels were written by an LLM and have not been human-reviewed, and the
gains seen on development queries were **only partly confirmed** on the held-out test queries.

## 1. What was built

| Component | Module (`src/neural_debugger/ml/retrieval/`) | Notes |
|---|---|---|
| Sentence-transformer dense retriever | `encoders.py`, `dense.py` | pinned HF revisions, per-model prefixes/max length, device `auto` (CUDA > MPS > CPU) with CPU fallback |
| Batch embedding + persistent store | `embed.py`, `embedding_store.py` | `data/embeddings/<model>/{embeddings.npy,meta.json}`; reused only if corpus hash, model id/revision, prefixes, max length and text template all match, else rebuilt |
| Exact vector index (+ framework sub-indexes) | `vector_index.py` | NumPy exact search and FAISS `IndexFlatIP`; see section 8 for why `auto` usually picks NumPy |
| Hybrid retrieval with RRF | `fusion.py` | weighted reciprocal-rank fusion over section rankings, deterministic tie-breaks |
| Cross-encoder reranker | `reranker.py` | rerank top-N of any retriever; `replace` or `rrf` mode |
| Framework awareness | `framework.py` + the `framework=` filter on every retriever | explicit filter (the UI selector) plus an optional corpus-trained detector/router |
| Named, reproducible systems | `systems.py`, `configs/neural.yaml` | every system is a config entry; pinned model revisions |
| Evaluation | `ml/evaluation/{system_eval,compare,qrels,tune_fusion,freeze,protocol}.py` | per-process runs (memory), paired bootstrap, judged-coverage metrics, freeze + test-access guard |

Try it: `uv sync --extra data --extra neural && make demo-neural Q="your error"` (first run downloads
the ~67 MB model; embeddings for the corpus are committed).

## 2. Models, licences, and deployment notes

| Role | Model (pinned revision in `configs/neural.yaml`) | Weights file | Licence | Used as |
|---|---|---|---|---|
| Embeddings (selected) | `thenlper/gte-small` | 67 MB | MIT | production dense encoder |
| Embeddings (compared) | `BAAI/bge-small-en-v1.5` | 134 MB | MIT | dev comparison |
| Embeddings (compared) | `sentence-transformers/all-MiniLM-L6-v2` | 91 MB | Apache-2.0 | dev comparison |
| Cross-encoder (selected) | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 134 MB | Apache-2.0 | optional reranker |
| Cross-encoder (compared) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 91 MB | Apache-2.0 | dev comparison |

* **Why small models.** Total download for everything compared was about 520 MB (plus PyTorch).
  Larger candidates (`bge-base` 438 MB, `bge-reranker-base` 1.1 GB, `bge-reranker-v2-m3` 2.3 GB) were
  deliberately *not* downloaded: the corpus is 5k chunks and a laptop CPU, and the measured
  difference between the three small embedding models was already within noise (section 5). They
  are the obvious next experiment if quality, not cost, becomes the constraint.
* **Licences are permissive (MIT / Apache-2.0)** as stated on the model cards (read from the Hugging
  Face API on 2026-10-25). Caveat that the licence field does not settle: the two cross-encoders were
  trained on **MS MARCO**, whose dataset terms restrict use to non-commercial research. Whether that
  carries over to model weights is a legal question this project has not resolved; the reranker is
  therefore optional and **off by default** for any commercial deployment until that is checked. The
  embedding models' training data (web-scale sentence pairs) has its own provenance caveats.
* **Deployment.** First use needs network access to huggingface.co (or pre-populate the cache); the
  pinned revisions make downloads reproducible. Running 100% offline afterwards is supported. Memory
  is about 520 MB RSS for dense/hybrid search and about 700-1,100 MB peak with the reranker (section 6).
* Documents and queries are processed locally; nothing is sent to a third party at query time.

## 3. Method (and why the numbers can be trusted, within limits)

1. **Development vs test.** Every choice (embedding model, RRF parameters, reranker, router) was made
   on the `train` + `val` queries only (138 authored, 23 FAQ). The test split stayed behind a guard
   (`--final-test`) that appends each access to `data/eval/results/test_access_log.jsonl`. After
   selection the configuration was frozen (`data/eval/m3_freeze.json`, `make freeze-verify`), then the
   four frozen systems were scored on test exactly once (plus once more with expanded judgments).
2. **Incomplete judgments.** The original labels were pooled from BM25's top-8 only. For a neural
   system, 46-65 per cent of its top-10 sections were never judged, so scoring it against the
   original labels under-counts it (the bias favours BM25). I therefore ran two **pool-expansion
   rounds on dev**: the top-5 sections of every candidate system (3 dense models, 3 hybrids, tuned
   hybrid, rerankers, routers) that no one had judged were read and judged by the same assistant that
   wrote the original labels (1,094 candidates; 77 judged relevant; `data/eval/pooled_judgments_neural.yaml`).
   For the **test** split, once the systems were frozen, the same procedure was applied to the four
   frozen systems' top-5 (118 candidates; 10 relevant; `pooled_judgments_neural_test.yaml`) - after
   seeing the candidates but not tuning anything on them. Test pooling is therefore disclosed
   post-hoc work; the original-label test numbers are reported too and are the untouched ones.
3. **Three ways to score, all reported.** (a) *original* judgments (frozen M2 labels), (b) *expanded*
   judgments (original + neural pools), (c) *condensed* NDCG@10 (unjudged sections removed from the
   ranking, Sakai 2007). `unjudged@10` is the share of a system's top-10 that has never been judged:
   the lower it is, the more the metric can be trusted. Even after expansion it is 8-30 per cent on dev and 17-35 per cent on test, so
   absolute numbers remain approximate.
4. **Paired statistics.** Differences between systems are paired per query with a 95 per cent
   bootstrap interval; the verbatim-FAQ slice (query = indexed heading) is reported separately
   because it is saturated by construction.
5. **Selection rules were declared before looking** and are in the tool docstrings: embedding model =
   best dense dev NDCG@10; fusion = a setting that beats the default (k=60, 1:1) with a paired
   interval excluding zero (best of those); reranker = best dev variant; router kept only if it helps.

## 4. Results on development queries (train + val)

Authored queries (n=138), final expanded judgments; interval = 95% bootstrap on NDCG@10:

| system | n | R@1 | R@5 | R@10 | MRR | NDCG@10 | unjudged@10 | condensed NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 138 | 0.231 | 0.521 | 0.638 | 0.656 | 0.552 [0.49, 0.61] | 0.14 | 0.575 |
| dense-minilm-l6 | 138 | 0.217 | 0.552 | 0.682 | 0.667 | 0.574 [0.52, 0.62] | 0.30 | 0.643 |
| dense-bge-small | 138 | 0.274 | 0.635 | 0.723 | 0.748 | 0.651 [0.60, 0.70] | 0.24 | 0.685 |
| dense-gte-small | 138 | 0.285 | 0.625 | 0.737 | 0.750 | 0.666 [0.61, 0.72] | 0.23 | 0.696 |
| hybrid-gte-small | 138 | 0.267 | 0.613 | 0.746 | 0.738 | 0.652 [0.60, 0.70] | 0.15 | 0.677 |
| hybrid-gte-tuned | 138 | 0.273 | 0.637 | 0.775 | 0.753 | 0.680 [0.63, 0.73] | 0.10 | 0.690 |
| rerank-l6-n30-rrf | 138 | 0.264 | 0.659 | 0.769 | 0.744 | 0.669 [0.62, 0.72] | 0.08 | 0.679 |
| rerank-l12-n30-rrf | 138 | 0.283 | 0.667 | 0.782 | 0.768 | 0.685 [0.63, 0.73] | 0.08 | 0.695 |
| rerank-l12-n30 | 138 | 0.265 | 0.595 | 0.712 | 0.736 | 0.624 [0.57, 0.68] | 0.18 | 0.649 |
| route-hard-99 | 138 | 0.268 | 0.619 | 0.760 | 0.735 | 0.664 [0.61, 0.72] | 0.13 | 0.676 |
| route-soft-1 | 138 | 0.254 | 0.603 | 0.746 | 0.710 | 0.643 [0.59, 0.70] | 0.14 | 0.661 |

Verbatim FAQ queries (n=23): **identical for every system** (saturated):

| system | n | R@1 | R@5 | R@10 | MRR | NDCG@10 | unjudged@10 | condensed NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| dense-minilm-l6 | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| dense-bge-small | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| dense-gte-small | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| hybrid-gte-small | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| hybrid-gte-tuned | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| rerank-l6-n30-rrf | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| rerank-l12-n30-rrf | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| rerank-l12-n30 | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| route-hard-99 | 23 | 0.957 | 1.000 | 1.000 | 0.978 | 0.984 [0.95, 1.00] | 0.90 | 1.000 |
| route-soft-1 | 23 | 0.870 | 0.957 | 1.000 | 0.917 | 0.937 [0.86, 1.00] | 0.90 | 1.000 |

Paired NDCG@10 differences (dev, authored, expanded judgments; wins/losses/ties per query):

| comparison | mean difference [95% CI] | W/L/T |
|---|---|---|
| dense (gte-small) vs BM25 | +0.114 [+0.059, +0.173] | 77/40/21 |
| hybrid, default fusion vs BM25 | +0.100 [+0.069, +0.133] | 80/17/41 |
| **hybrid, tuned fusion vs BM25** | **+0.128 [+0.090, +0.169]** | 79/23/36 |
| hybrid + reranker (L12, rrf) vs BM25 | +0.133 [+0.096, +0.173] | 81/29/28 |
| hybrid + reranker (L12, rrf) vs tuned hybrid | +0.005 [-0.018, +0.026] | 46/41/51 |
| dense vs tuned hybrid | -0.014 [-0.042, +0.015] | 42/53/43 |
| reranker in `replace` mode vs tuned hybrid | -0.056 [-0.096, -0.017] | 44/59/35 |
| hard framework router vs tuned hybrid | -0.017 [-0.044, +0.005] | 8/12/118 |
| soft framework router vs tuned hybrid | -0.037 [-0.073, -0.008] | 17/14/107 |
| bge-small vs gte-small (dense) | -0.015 [-0.046, +0.015] | 41/48/49 |
| MiniLM-L6 vs gte-small (dense) | -0.092 [-0.141, -0.047] | 36/71/31 |

What the development study says:

* Dense and hybrid retrieval **clearly beat BM25** (about +0.10 to +0.13 NDCG@10). The hybrid is the
  *robust* win: it loses to BM25 on only 17-23 of 138 queries, versus 40 for dense alone.
* `gte-small` vs `bge-small`: not distinguishable; `all-MiniLM-L6` is clearly worse. `gte-small` was
  selected by the declared rule (best dense dev score) and is also the smallest.
* Fusion tuning (k=10, dense weight 1.5; `data/eval/results/m3/fusion_tuning_dev.json`) gave +0.028
  over the default on dev (CI [+0.008, +0.049]). It is the best of 20 tried settings, so it is
  optimistic by construction; the surrounding grid is a smooth plateau, not an isolated spike.
* The off-the-shelf **MS MARCO cross-encoder does not help on its own**: ordering by its score alone
  *hurts* (-0.056), and fusing it with the hybrid ranking only ties the hybrid (+0.005, CI spans zero)
  at 7-12x the latency. It is a general web-passage model, not trained on API documentation.
* **Framework routing does not help.** The corpus-trained detector is 90.6 per cent accurate on dev
  queries and its mistakes cost more than the confusion it removes. An *oracle* filter (perfect
  framework knowledge) changes NDCG@10 by +0.014 for BM25, +0.002 for dense, and -0.002 for the hybrid:
  **dense retrieval already resolves most cross-framework confusion**, so there is little left to gain.
  The user-facing framework selector is still supported (and always overrides the router).

## 5. Final evaluation on the held-out test split (frozen systems, evaluated once)

Authored queries (n=32). Under the **original (frozen M2) judgments**:

| system | n | R@1 | R@5 | R@10 | MRR | NDCG@10 | unjudged@10 | condensed NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 32 | 0.329 | 0.567 | 0.692 | 0.666 | 0.597 [0.47, 0.72] | 0.27 | 0.684 |
| dense-gte-small | 32 | 0.251 | 0.581 | 0.698 | 0.608 | 0.580 [0.45, 0.70] | 0.65 | 0.757 |
| hybrid-gte-tuned | 32 | 0.256 | 0.672 | 0.725 | 0.636 | 0.601 [0.48, 0.72] | 0.49 | 0.746 |
| rerank-l12-n30-rrf | 32 | 0.360 | 0.628 | 0.766 | 0.709 | 0.653 [0.53, 0.76] | 0.46 | 0.760 |

Under **expanded judgments** (original + neural pool review of the frozen systems' top-5):

| system | n | R@1 | R@5 | R@10 | MRR | NDCG@10 | unjudged@10 | condensed NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 32 | 0.315 | 0.525 | 0.645 | 0.666 | 0.569 [0.44, 0.69] | 0.18 | 0.622 |
| dense-gte-small | 32 | 0.242 | 0.610 | 0.708 | 0.647 | 0.604 [0.48, 0.72] | 0.35 | 0.678 |
| hybrid-gte-tuned | 32 | 0.249 | 0.663 | 0.753 | 0.670 | 0.626 [0.51, 0.74] | 0.17 | 0.658 |
| rerank-l12-n30-rrf | 32 | 0.343 | 0.625 | 0.784 | 0.723 | 0.666 [0.55, 0.77] | 0.19 | 0.692 |

Paired NDCG@10 differences on test (authored, n=32):

| comparison | original judgments | expanded judgments |
|---|---|---|
| dense vs BM25 | -0.017 [-0.147, +0.125] | +0.035 [-0.099, +0.179] |
| tuned hybrid vs BM25 | +0.003 [-0.088, +0.104] | +0.058 [-0.040, +0.165] |
| hybrid + reranker vs BM25 | +0.055 [-0.025, +0.146] | **+0.097 [+0.013, +0.190]** |
| hybrid + reranker vs tuned hybrid | - | +0.039 [-0.020, +0.105] |

Verbatim FAQ (n=10, original judgments): BM25 NDCG@10 0.963, every neural system 1.000 (paired
difference +0.037, CI [0.000, +0.111]; 1 query differs). This slice is lexically trivial and is not
evidence of anything beyond "headings are easy to find".

Reading the test results honestly:

* **Direction is consistent with dev, size is smaller, and almost nothing is statistically
  significant.** With 32 queries the intervals are about +/-0.1. Only "hybrid + reranker vs BM25"
  under expanded judgments excludes zero.
* The judgment problem is visible: on the original labels the neural systems have 46-65 per cent of
  their top-10 unjudged and score *no better than BM25*; on **condensed** lists (unjudged removed) they
  lead (0.746-0.760 vs 0.684), and after expansion they lead on the plain metric too. The truth is
  most likely between the original and expanded numbers.
* Dev gains (+0.13) did not fully carry over to test (+0.06 expanded). Part of that is selection
  optimism (best-of-20 fusion setting, best-of-N reranker), part is the tiny test set.
* The reranked hybrid is the best system on test, but its dev advantage over the plain hybrid was nil
  (+0.005); on test it is +0.039 (CI spans zero). I would not claim the reranker helps; I would say
  the data cannot rule it out and that it costs 10x the latency.

Test cost columns are in section 6. The test-access log shows the four frozen systems were scored
on test exactly twice each (original, then expanded judgments).

## 6. Cost: indexing time, latency, memory (Apple M4 Max, macOS, Python 3.12, one process per system)

| system | device | build / load (s) | embed (s, one-off) | p50 / p95 query (ms) | RSS after build (MB) | peak RSS (MB) | extra storage (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | cpu | 0.8 | 0.0 | 0.8 / 1.4 | 161 | 162 | 0.0 |
| dense-minilm-l6 | mps | 6.8 | 5.8 | 6.7 / 7.9 | 515 | 639 | 7.7 |
| dense-bge-small | mps | 6.7 | 11.3 | 10.7 / 11.8 | 519 | 636 | 7.7 |
| dense-gte-small | mps | 6.6 | 10.7 | 10.0 / 11.7 | 516 | 642 | 7.7 |
| dense-gte-small | cpu | 6.3 | 10.7 | 10.2 / 15.2 | 516 | 567 | 7.7 |
| hybrid-gte-small | mps | 7.3 | 10.7 | 11.2 / 13.2 | 529 | 654 | 7.7 |
| hybrid-gte-tuned | mps | 7.1 | 10.7 | 11.1 / 12.8 | 526 | 652 | 7.7 |
| hybrid-gte-tuned | cpu | 7.1 | 10.7 | 11.6 / 16.4 | 528 | 580 | 7.7 |
| rerank-l6-n30-rrf | mps | 7.5 | 10.7 | 64.3 / 76.2 | 538 | 786 | 7.7 |
| rerank-l12-n30-rrf | mps | 7.5 | 10.7 | 105.7 / 123.0 | 538 | 786 | 7.7 |
| rerank-l12-n30-rrf | cpu | 7.4 | 10.7 | 460.7 / 575.4 | 539 | 1124 | 7.7 |
| rerank-l12-n30 | mps | 7.5 | 10.7 | 106.0 / 129.8 | 536 | 785 | 7.7 |
| route-hard-99 | mps | 7.9 | 10.7 | 9.1 / 12.4 | 530 | 656 | 7.7 |
| route-soft-1 | mps | 8.4 | 10.7 | 10.6 / 13.3 | 528 | 654 | 7.7 |

* Embedding the 4,995 chunks takes 6-11 s on MPS, one-off per model/corpus version (on CPU, MiniLM
  encoded a 600-chunk sample in 1.8 s); the vectors are 7.7 MB (float32, 384-d).
* "build / load" is wall time to construct the system in-process (model load + index); BM25 alone is
  0.8 s. RSS after build is dominated by the PyTorch runtime and model (BM25 alone: 161 MB).
* **CPU-only works and gives identical metrics** (rows marked `cpu` match the MPS rows to the
  third decimal): dense/hybrid p50 10-12 ms, but reranking on CPU is about 460 ms/query vs 106 ms on MPS.
* Latency is in-process, one query at a time, 3 warm-up queries excluded; it says nothing about a
  deployed service under load (Milestone 7).

## 7. Error analysis: which queries improved or regressed

Tuned hybrid vs BM25 on dev authored queries (n=138): **79 improved, 23 regressed, 36 tied**
(`scripts/error_analysis.py data/eval/results/m3/dev_final bm25 hybrid-gte-tuned`).

Improvements cluster in the documented BM25 weaknesses:

| Weakness | Example (BM25 rank -> hybrid rank of first relevant section) |
|---|---|
| Error text absent from the explaining page (vocabulary mismatch) | `ValueError: not enough values to unpack (expected 2, got 1)` -> *tuples and sequences*: not retrieved -> rank 1; `Trying to backward through the graph a second time` -> `Tensor.backward`: rank 12 -> 2 |
| Natural-language paraphrase | `what is the difference between is and == when comparing objects` -> Python FAQ on identity: not retrieved -> rank 2; `guarantee cleanup code runs even if an exception happens` -> *clean-up actions*: not retrieved -> rank 3 |
| Cross-framework confusion | FastAPI `RuntimeError: Directory 'static' does not exist` -> static-files docs: rank 14 -> 1 (BM25 returned Python's *exception chaining*) |
| Short, concept-style questions | `dataclass field that defaults to an empty list` -> `dataclasses.field`: rank 3 -> 1; `send a large generated response chunk by chunk` -> streaming docs: 4 -> 1 |

Regressions are mostly BM25's strengths: exact identifiers or API names that dense similarity blurs.

| Regression | What happened |
|---|---|
| `RuntimeError: Function 'LogBackward0' returned nan values` (-0.26) | the hybrid ranked `logging.LogRecord` (similar to the "Log..." token) first, while BM25 ranked the anomaly-detection docs first |
| `format a datetime object as a readable string` (-0.36) | top hit became `time.__format__` instead of `datetime.strftime`: near-synonym API pages compete |
| `mount an existing Flask application inside my FastAPI app` (-0.49) | FastAPI's generic sub-application (mount) docs outranked the WSGI/Flask section |
| `ignore one specific exception without writing try except pass` (-0.33) | `contextlib.suppress` stayed out of the top 10 for both; BM25 had it at 7, hybrid at 14 |
| `swagger docs page ... nginx under /api prefix` (-0.17) | both retrieve *metadata/openapi-url* first; the *root_path* section slipped from 2 to 6 |

Some "regressions" are metric artefacts: for `turn off dropout and batch norm updates...` both systems
return the relevant section at rank 1, but the expanded judgments add relevant sections that BM25
happened to rank earlier. And some of BM25's wins are lucky exact matches rather than understanding.

## 8. Engineering findings worth knowing

* **FAISS is implemented but usually not the chosen backend.** On this corpus exact NumPy search takes
  0.06 ms/query vs 0.14 ms for FAISS `IndexFlatIP` (4,995 x 384, in separate processes), so FAISS gives
  no speed-up at this scale. Worse, on macOS FAISS and PyTorch each bundle a `libomp`, and the process
  **aborts (`OMP: Error #15`) once both are loaded and FAISS searches**, which would break the CPU
  fallback. `backend="auto"` therefore uses FAISS only when it is safe (no PyTorch in the process, or
  not macOS) and otherwise NumPy; FAISS is tested for equivalence in a subprocess without PyTorch.
  Switch to FAISS (HNSW/IVF) when the corpus is 100x larger, preferably in a separate process.
* **macOS hidden-`.pth` pitfall.** Files under a synced `~/Desktop/.venv` acquired the macOS `hidden`
  flag and Python 3.12.15 silently skips hidden `.pth` files, so the editable install stopped importing.
  The repo no longer depends on it: pytest sets `pythonpath = ["src"]` and `make` exports `PYTHONPATH`.
  If imports fail: `chflags -R nohidden .venv`.
* **Device parity.** Embeddings built on MPS and CPU agree (cosine >= 0.9999), CPU encoding is bitwise
  deterministic, and CPU/MPS runs produce the same ranking metrics.
* Dense retrieval embeds `title / heading path / content` (the same text BM25 indexes); texts longer
  than the model's limit (256 tokens for MiniLM, 512 for bge/gte) are truncated.

## 9. Limitations and what is not claimed

1. **Labels are LLM-authored and unreviewed by a human**, and pooling was done by the same assistant
   (`docs/evaluation.md`). Neural pool expansion reduces, but does not remove, the BM25 pooling bias:
   8-35 per cent of top-10 sections are still unjudged. Human review of a sample is still the main
   outstanding quality step.
2. **Small test set** (32 authored queries): most comparisons are inconclusive. Do not read a ranking
   between dense, hybrid and reranked from the test table alone; combine the dev and test evidence.
3. **Selection optimism.** Fusion parameters were picked from 20 settings and the reranker from 4
   variants on dev; dev numbers are biased upward.
4. **Post-hoc test pooling** (see section 3). The original-label test results are untouched by it.
5. **Not evaluated:** larger models, domain fine-tuning (the custom-trained encoder from the original
   brief is still ahead), query-side generation (HyDE), real-user queries, other corpora/versions.
6. The reranker's training-data licence (MS MARCO) is unresolved for commercial use.
7. A strong claim like "neural retrieval fixes BM25's weaknesses" is **not** supported; the supported
   claim is "on this corpus, dense/hybrid retrieval is better than BM25 by a margin that is clear on
   138 dev queries and directionally but not significantly confirmed on 32 test queries".

## 10. Reproduce

```bash
uv sync --extra data --extra neural
make embed M=gte-small                       # (re)generate embeddings; skipped if up to date
make eval-dev S=hybrid-gte-tuned             # dev (train+val), expanded judgments
make freeze-verify                           # M2 baseline + M3 configuration unchanged
uv run python -m neural_debugger.ml.evaluation.compare --dir data/eval/results/m3/dev_final \
    --reference bm25 --systems dense-gte-small hybrid-gte-tuned rerank-l12-n30-rrf
```
