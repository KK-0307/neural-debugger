# Progress

**Current milestone:** 5 / 5.1 - RAG with validated citations, live-LLM verification, semantic grounding and a
deployable image. **Live generation was exercised with one real model** (Gemini 3.1 Flash-Lite, free tier,
2026-10-26): 51 questions, valid JSON every time, no invalid citation ids. **The honest headline is mixed:** it answered
80% of my own in-scope questions but **only 9 of 24 independently generated ones** (the rest are retrieval misses,
over-abstention on verbose questions, and troubleshooting causes the reference docs do not contain). **Docker is verified
only by a local simulation; the image has never been built here.** No human labelled anything. Awaiting your review; not
proceeding to Milestone 6.

**Milestone 5.1 in one paragraph (details: `docs/rag.md` section 11, `docs/semantic-grounding.md`).** Added a free hosted
LLM path (Gemini through the existing OpenAI-compatible provider; env-var key, hard request cap, spacing, no fallback model),
a rate-limited live acceptance harness and an offline replay/re-score tool; a semantic grounding stage (rules for numbers,
signatures, identifiers, negation; optional NLI) with a deterministic synthetic adversarial benchmark (lexical alone silently
accepts 52% of defective claims, rules 21%, rules + NLI 2.7% on the test half; thresholds chosen on dev only; real-model
recall unmeasured); 30 model-generated independent questions; and a Docker image with the retrieval stack, CPU-only
PyTorch, corpus, embeddings and a baked offline embedding model, plus a CI smoke test. Real-model runs exposed 5 defects
that no mocked test had (`e.g.` parsed as an identifier, dotted names, user-typed URLs stripped, a first semantic design
that flagged 85-95% of claims and removed five correct ones, a retired model that is still listed); all fixed and tested.

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
| 5 | RAG with validated citations (API, providers, evaluation) | **Done**; live run on one free-tier model (51 questions, agent-reviewed only) |
| 5.1 | Live verification, semantic grounding, Docker retrieval image | **Done locally; Docker image build/size verified only in CI** (see below) |
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
| `uv run pytest` (full env, models cached) | **286 passed, 1 skipped** (the skip is the live Anthropic test; M5.1 added 35 tests: semantic stage, free-tier config, harness, Dockerfile guards, freeze) |
| CI-equivalent env (`uv sync --locked --extra data --extra index`, no PyTorch; validates the regenerated `uv.lock`) | ruff + format clean, strict mypy clean (98 src / 41 test files), **244 passed, 9 skipped**; GitHub Actions for this milestone's commit: see the final report |
| `make freeze-verify` (M2 baseline + M3 + M4 + M5 + new M5.1 evidence) | intact; frozen M5 evaluation **re-run with the 5.1 code: 146 of 146 non-timing numbers identical** |
| Frontend `lint`, `typecheck`, `npm test` (8), `npm run build` | clean (frontend untouched) |
| Container simulation (no Docker here): staging dir with exactly the Dockerfile's `COPY` set, fresh `uv sync --frozen --extra neural`, model baked into an isolated offline `HF_HOME`, scrubbed env, mock OpenAI-compatible server | `/api/v1/debug` -> `answered`, 4 PyTorch citations; without a provider -> informative 503 `llm_not_configured` listing 5 retrieved sources; first request 31 s (cold retriever) |
| Secrets audit | the API key value appears in no file in the repository or in any run output; `.env` is git-ignored; a key-shaped test fixture is now assembled at runtime so secret scanners do not trip |
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
- Retrieval is the weak link of the whole pipeline: all 3 false abstentions on my own questions were retrieval misses
  (right section ranked 17th and 36th, or absent), and 6 of 15 in-scope abstentions on the independent set came from a
  lexical gate that over-abstains on verbose questions. Fixing that is Milestone 2-4 territory and was not attempted.
- The corpus is reference documentation, not a troubleshooting knowledge base: "why does X happen" questions often cannot
  be answered from it, and the system (correctly) refuses.
- Semantic grounding recall on real model errors is unmeasured (no known defective claim in 98 real claims); `disputed`
  flags were false alarms on correct claims 2-3 times in 98 and lower the answer's confidence.
- Free-tier Gemini is unreliable under load (Google 503 "high demand" for 8 of 51 first-pass questions); limits are
  unpublished; on the free tier Google may use prompts to improve its products.
- Caveats can be noisy (seven version notes on one answer). The first request loads the retriever (6-31 s).
- Docker: Dockerfile, Linux CPU wheels, image size and the CI smoke test have not run yet. Only the simulation above did.

## Milestone 5 - live verification: closed, with caveats

Closed on 2026-10-26 with a **free-tier Gemini model** (no Ollama, no local weights, no paid request). Measured: JSON
compliance, citation validity, abstention, latency, tokens (`docs/rag.md` section 11). **Still not measured:** the
Anthropic provider against the real API, any other model, load, a meaningful prompt-injection sample (only 2 poisoned-
document cases reached a live model), the model-graded evaluation rows (deliberately not produced: a second LLM is not
ground truth), and anything human-verified (n = 0). Every correctness judgement of an answer is an **agent review**.

## Remaining work (needs your direction)

1. Human review of a label sample (and optionally Stack Overflow-derived labels, which need approval).
2. Decide whether to invest further in the custom model (more/LLM-generated training data, distillation
   from a pretrained model) or use the pretrained hybrid as the retrieval backbone for RAG.
3. Decide whether to attack retrieval recall and the gate's over-abstention on verbose questions before the UI, and whether
   to add a troubleshooting-style source (needs your approval, e.g. Stack Overflow) to the corpus.
4. Human review of a sample of live answers and of the semantic flags; the debugging UI and persistence (Milestone 6).
