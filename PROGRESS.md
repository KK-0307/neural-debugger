# Progress

**Current milestone:** 5 - RAG with validated citations: **implemented, tested and evaluated offline; live
generation with a real LLM is BLOCKED (no credentials) and therefore UNVERIFIED, so the milestone's "works end
to end with a configured real LLM provider" criterion is NOT met.** Awaiting your review; not proceeding to
Milestone 6.

**Milestone 5 in one paragraph (details: `docs/rag.md`).** `neural_debugger.rag` implements question
preparation, retrieval with the pretrained hybrid (custom transformer selectable), evidence selection and
sanitisation, a calibrated pre-LLM abstention gate, a provider interface (Anthropic Messages and
OpenAI-compatible HTTP providers + deterministic test providers), JSON-contract prompting with one repair
retry, citation *validity* and (separately) *support* checks, version/conflict caveats, injection defences,
and `POST /api/v1/debug`. 97 new tests. Offline evaluation (dev queries only, extractive test provider, so
plumbing not LLM quality): pre-LLM gate abstains on 84% of 32 out-of-scope questions (81% on the held-out half)
with 3.7% false abstentions; fault injection: fabricated citations / uncited claims / unsupported identifiers /
foreign links 0 of 132 survive, inverted-meaning claims 132 of 132 survive. Model-graded and human-verified
evaluation rows are empty.

**Milestone 4 (approved) in one paragraph (details: `docs/ml-model.md`).** A from-scratch transformer (0.58M
parameters, mean-pooled, L2-normalised, shared towers) trained with InfoNCE on synthetic documentation
pairs does **not** beat BM25: test NDCG@10 (authored, n=32, M4 labels) is 0.314 dense / 0.531 hybrid vs
BM25 0.560 and the pretrained hybrid 0.621. Smaller models were better (0.58M > 1.5M > 5.3M); BM25 hard
negatives hurt; adding real train queries helped on val but not on test. Its advantages are size (2.3 MB)
and speed (~1 ms/query). Test pool expansion for the custom systems was post-hoc and is disclosed.

> **Roadmap note.** Milestone 3 was redefined at your request from "custom PyTorch transformer" to
> "neural retrieval" built on *pretrained* models; the custom transformer from the original brief was
> built in Milestone 4 and compared against those baselines.

| # | Milestone | Status |
|---|---|---|
| 1 | Project foundation | Done (Docker builds and CI runs not yet verified) |
| 2 | Documentation retrieval (corpus, labels, BM25) | Done, **frozen** (labels LLM-authored; agent second-pass review only) |
| 3 | Neural retrieval (dense, hybrid, rerank, framework filter) | **Done**; results below, with caveats |
| 4 | Custom transformer + contrastive training | **Done**; negative result vs. baselines (`docs/ml-model.md`) |
| 5 | RAG with validated citations (API, providers, evaluation) | **Implemented; live LLM generation unverified (blocked)** |
| - | Debugging UI, persistence, performance, beta | Not started |

## Step 1 - Milestone 2 finalised and committed

- 38 dev queries / 102 judgments re-read by the same assistant (agent second pass; **not** independent,
  **not** human): 71 confirmed, 23 marginal, 6 uncertain, 2 weak; 6 queries flagged
  (`data/eval/label_review.yaml`). Labels not edited; test-split labels not reviewed.
- BM25 configuration, splits, corpus and results frozen (`baseline_freeze.json`, `make freeze-verify`).
- Test split guarded (`--final-test`) with an append-only access log; M2 BM25 test run recorded
  retroactively.
- Local commit `8f898a9` (not pushed).

## Step 2-3 - Milestone 3 deliverables

| Requested | Delivered | Where |
|---|---|---|
| Sentence-transformer dense retriever | `gte-small` (selected on dev from 3 candidates) | `ml/retrieval/{encoders,dense}.py` |
| Batch embedding of all 4,995 chunks | 11 s on MPS; CPU fallback verified identical | `ml/retrieval/embed.py` |
| Persistent embeddings + efficient index | versioned store (rebuild on any fingerprint change); exact index with per-framework sub-indexes; FAISS implemented but NumPy is used when FAISS is unsafe/slower (see findings) | `embedding_store.py`, `vector_index.py` |
| Hybrid BM25 + dense with RRF | weighted RRF, tuned on dev | `fusion.py` |
| Optional cross-encoder reranking | L6 and L12 MS-MARCO cross-encoders, replace / rrf modes | `reranker.py` |
| Framework-aware filtering | explicit filter on every retriever + corpus-trained detector/router (rejected: no benefit) | `framework.py` |
| Reproducible configs / versions / determinism | pinned HF revisions in `configs/neural.yaml`, embedding fingerprints, frozen M3 manifest, CPU bitwise deterministic | `systems.py`, `freeze.py` |

### Results (NDCG@10, authored queries; details, CIs and all metrics in `docs/neural-retrieval.md`)

| | dev n=138 (expanded judgments) | test n=32 original labels | test n=32 expanded labels |
|---|---:|---:|---:|
| BM25 | 0.552 | 0.597 | 0.569 |
| Dense (gte-small) | 0.666 | 0.580 | 0.604 |
| Hybrid (tuned RRF) | 0.680 | 0.601 | 0.626 |
| Hybrid + cross-encoder (L12, rrf) | 0.685 | 0.653 | 0.666 |

- **Dev:** dense/hybrid beat BM25 significantly (hybrid +0.128, 95% CI [+0.090, +0.169]; 79 queries
  better, 23 worse). The cross-encoder adds nothing significant over the hybrid (+0.005, CI spans 0).
- **Test (final, scored once per judgment set):** same direction, smaller gains; only hybrid+rerank vs
  BM25 under expanded judgments excludes zero (+0.097, CI [+0.013, +0.190]).
- **Verbatim FAQ queries** (query = indexed heading): saturated; every system ~1.0.
- Cost (M4 Max): index build 0.8 s (BM25) / ~7 s incl. model load (hybrid); p50 query 0.8 ms (BM25),
  10-11 ms (dense/hybrid), 106 ms (+reranker, MPS) / 460 ms (CPU); RSS 161 MB (BM25) vs ~520 MB (neural),
  up to ~1.1 GB peak with reranker on CPU; embeddings 7.7 MB.

### Incomplete judgments (handled explicitly)

Under the original BM25-pooled labels 46-65% of the neural systems' top-10 sections had never been
judged. Two pool-expansion rounds on dev (1,094 candidates read, 77 relevant) and, after freezing, one on
test (118 read, 10 relevant) were done by the assistant (not human-verified). Every table reports
`unjudged@10`, original/expanded judgments, and condensed-list NDCG. Test pooling was done after the
frozen systems' rankings existed (disclosed; original-label test results are unaffected by it).

### Error analysis

`docs/neural-retrieval.md` section 7: tuned hybrid vs BM25 improved 79 / regressed 23 / tied 36 dev
queries; improvements concentrate in vocabulary mismatch, natural-language paraphrase and
cross-framework confusion; regressions are mostly exact-identifier cases where BM25 was lucky or
precise (e.g. `LogBackward0`, `datetime.strftime`).

### Findings that change how the system should be used

- **FAISS + PyTorch abort on macOS** (duplicate `libomp`) and NumPy exact search is faster at this
  scale, so the default backend is NumPy unless FAISS is safe; FAISS is implemented and tested.
- **Framework routing did not help** (even a perfect framework filter adds <= +0.014): dense retrieval
  already handles most cross-framework confusion; the user-selected framework filter is kept.
- **The reranker is optional and off by default:** no significant gain, 10x latency, MS MARCO licence caveat.
- macOS can hide `.venv` files and Python then skips the editable `.pth`; the repo no longer depends on it.

## Verification actually performed

| Check | Result |
|---|---|
| `uv run pytest` (full env, models cached) | **251 passed, 1 skipped** (M4: +37; M5: +97 collected, of which the live-LLM test is the 1 skip) |
| CI-equivalent env (`--extra data --extra index`, no PyTorch) | ruff clean, strict mypy clean (93 src / 37 test files), **209 passed, 9 skipped** (model/PyTorch/live-LLM tests); GitHub Actions python, frontend and docker jobs green on M5 `bd2cb61` |
| `make freeze-verify` (M2 baseline + M3 + M4 + M5 evaluation) | intact |
| Frontend `lint`, `typecheck`, `npm test` (8) | clean (frontend untouched in M5; `npm run build` left to CI) |
| Core-only environment (the Docker backend image's dependency set) | app imports; `/api/v1/debug` answers 503 `retrieval_unavailable` instead of crashing |
| Wheel build + clean-venv install + `/health` on real PostgreSQL 16 | 200 `ok`, then 503 after stopping Postgres; retrieval modules import from the wheel without PyTorch |
| CPU vs MPS | identical ranking metrics; embeddings cosine >= 0.9999; CPU bitwise deterministic |
| Test access log | M3: 4 systems x 2 judgment sets = 8; M4: 6 systems x 2 judgment sets = 12; + 1 retroactive M2 entry (21 lines); asserted by a test |

## Known issues / limitations

- **Labels are not human-verified**; pooling and expansion were done by the same assistant. Gains are
  measured against these labels. Human review of a sample is the main outstanding quality step.
- **Test set is small** (32 authored): most comparisons inconclusive. Dev numbers carry selection
  optimism (best of 20 fusion settings, best of 4 rerank variants).
- Docker is not installed locally; CI (GitHub Actions, private repo) builds both images and is green as of
  the M3 push fixes.
- M4: no mid-run training resume; one tokenizer size / temperature; no LR sweep; test pool expansion for
  the custom systems was done after seeing test rankings; seed 0 weights are shipped, not the best seed.
- Larger models / fine-tuning / HyDE / real-user queries not evaluated.
- No database schema or API wiring for retrieval yet (RAG milestone).

## Milestone 5 - blocked item and how to unblock it

Live generation could not be verified: there is no `ANTHROPIC_API_KEY` (or other provider key or local LLM
runtime) in this environment, and I did not use paid APIs or the harness' own endpoint. To close the gap,
either export a key and run the three commands at the top of `docs/rag.md` section 2 (about a few cents), or
point `--provider openai_compatible --base-url http://localhost:11434/v1` at a local model server (free). Until
then the model-graded rows of the RAG evaluation, real abstention behaviour, JSON compliance and
prompt-injection resistance of an actual model are **unmeasured**.

## Remaining work (needs your direction)

1. Human review of a label sample (and optionally Stack Overflow-derived labels, which need approval).
2. Decide whether to invest further in the custom model (more/LLM-generated training data, distillation
   from a pretrained model) or use the pretrained hybrid as the retrieval backbone for RAG.
3. Live-LLM verification of Milestone 5 (above), then the debugging UI / persistence / retrieval extras in the Docker image.
