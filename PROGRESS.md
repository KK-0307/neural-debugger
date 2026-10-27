# Progress

**Current milestone:** 6 - the Neural Debugger application (with a focused coverage pass, 5.2, before it). **Built, tested and
used in a real browser against the real backend and the real Gemini free-tier model; nobody but me has used it, and no result here
is human-verified.** Awaiting your review; not proceeding to Milestone 7.

**Milestone 5.2 in one paragraph (details: `docs/rag-coverage.md`).** On a **held-out set of 48 questions written by Gemini and
frozen before any change** the share of in-scope questions answered rose from **11/36 (31%) to 14/36 (39%)**; all 12 out-of-scope
questions still abstained, citation ids stayed 100% valid, and agent-reviewed answer quality was 8 good / 1 generic / 2 weak
before and 10 / 1 / 3 after. **That gain is not statistically established** (4 questions gained, 1 lost, sign-test p = 0.375;
PyTorch got slightly worse). What produced it: a 4-page official-documentation **supplement** merged over the untouched base
corpus (needed because Python's `sys`, `sqlite3` and the 3.12 "what's new" were not indexed), **traceback API hints**, and an
**evidence profile that shows the model 7 sections while the sufficiency gate keeps judging exactly what it judged before**
(measured: letting the gate see the larger set cut out-of-scope abstention from 81% to ~70%, which the brief forbids). On the
development set the gate blocked none of the 13 answerable questions, so its thresholds were not touched; the held-out set later
showed it **does** over-abstain on verbose symptom questions (4 clear cases), recorded as the main remaining limitation and
deliberately not tuned on. Query normalisation, lexical re-scoring, query decomposition and cross-encoder reranking were tried and
rejected with numbers. GitHub issues were **not** ingested (no redistribution licence); Stack Overflow was not used.

**Milestone 6 in one paragraph (details: `docs/frontend.md`).** A responsive Next.js app: paste a trace or question, optionally
pick Python / PyTorch / FastAPI, get a status banner, likely cause, numbered fixes, copyable example code (never run), conflicts,
what the docs did not cover, per-statement badges, citation chips that open the exact text the model was shown, an honest "how sure
is this?" panel built only from validator counters, specific error states, local history, and a feedback form
(question stored **only on opt-in**, secrets redacted, append-only JSONL). New `POST /api/v1/feedback`; `/debug` now also returns
sources' excerpts, the gate and semantic fields. 30 new frontend tests + 9 backend tests; verified in the browser at desktop and
375 px, offline provider and a live Gemini answer (3.7 s).

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
| 5.1 | Live verification, semantic grounding, Docker retrieval image | **Done**; CI green on `d148b6d` incl. real image build + smoke tests |
| 5.2 | Coverage pass (diagnosis, supplement corpus, traceback hints, evidence profile) | **Done locally**; modest, not significant gain on held-out (`docs/rag-coverage.md`) |
| 6 | Debugging UI (paste, diagnose, cite, uncertainty, feedback) | **Done locally**; browser-verified by me only; not yet in CI |
| - | Persistence, performance, beta | Not started |

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
| `uv run pytest` (full env, models cached) | **304 passed, 1 skipped** (the skip is the live Anthropic test; 5.2 + 6 added 18) |
| CI-equivalent env (`uv sync --locked --extra data --extra index`, no PyTorch) | ruff + format clean, strict mypy clean (101 src / 43 test files), **262 passed, 9 skipped**; **not yet pushed, so CI has not run on Milestones 5.2/6** (the Docker image with the new `COPY` paths has not been built anywhere) |
| `make freeze-verify` (M2, M3, M4, M5, M5.1 and new M5.2 evidence) | all intact. The held-out question file's sha256 is identical to when it was generated, before any change. **Re-running the frozen M5 evaluation with today's code differs in 4 of 146 non-timing numbers** (one train answer low -> medium confidence); attributed by running the pushed 5.1 commit: the cause is a 5.1 change, 5.2 adds none (an earlier "146 of 146 identical" was true when checked and stale after the user-typed-identifier fix; docs corrected) |
| Frontend `lint`, `typecheck`, `npm test`, `npm run build` | clean; **38 tests** (8 older + 30 new) |
| UI in a real browser (Claude built-in browser) | example -> answer, chip -> source text, abstention view, feedback stored with `question: null`, history restore, 375 px with no overflow; a live Gemini answer through the UI in 3.7 s. One browser engine, two widths, no screen-reader or axe audit |
| **CI Docker job** (the first real image build) | image **2,766 MiB**; no `.env`/secret-like variable/key in the image; container smoke test with a provider: `answered`, 4 citations, first request 9.6 s; without a provider: 503 `llm_not_configured` with 5 sources. Local Docker is not installed, so everything about the image is CI evidence |
| Container simulation before CI (no Docker here): staging dir with exactly the Dockerfile's `COPY` set, fresh `uv sync --frozen --extra neural`, model baked into an isolated offline `HF_HOME`, scrubbed env, mock OpenAI-compatible server | `/api/v1/debug` -> `answered`, 4 PyTorch citations; without a provider -> informative 503 `llm_not_configured` listing 5 retrieved sources; first request 31 s (cold retriever) |
| Secrets audit | the API key value appears in no file in the repository or in any run output; `.env` is git-ignored; a key-shaped test fixture is now assembled at runtime so secret scanners do not trip |
| Wheel build + clean-venv install + `/health` on real PostgreSQL 16 | 200 `ok`, then 503 after stopping Postgres; retrieval modules import from the wheel without PyTorch |
| CPU vs MPS | identical ranking metrics; embeddings cosine >= 0.9999; CPU bitwise deterministic |
| Test access log | M3: 4 systems x 2 judgment sets = 8; M4: 6 systems x 2 judgment sets = 12; + 1 retroactive M2 entry (21 lines); asserted by a test |

## Known issues / limitations

- **Coverage (5.2):** only 39% of in-scope held-out questions are answered; the gate over-abstains on verbose symptom questions (4 clear cases); 2 of 13 development golds are still not retrieved; Python stdlib coverage is 31 modules; citation *relevance* is not guaranteed (a weakly related citation passed the checks on one held-out answer).
- **The held-out gain is within noise** (p = 0.375) and PyTorch regressed 3 -> 2; one good answer was lost when the model was shown more evidence.
- **UI (6):** unreviewed by any human, one browser engine, feedback is stored but nothing consumes it, no auth, no streaming, history is browser-local only.

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
- Docker image is **2.7 GiB** (CPU-only torch, sentence-transformers, scipy/sklearn, faiss, plus bytecode); no multi-stage slimming was attempted, and the CI guard (< 3 GiB) leaves little headroom. The NLI model (283 MB) is not in the image, so `ND_RAG_SEMANTIC_CHECK=nli` would download it at run time.

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
3. A filler-aware coverage score for the gate, calibrated on a **new** development set (the held-out one is spent) and checked
   against out-of-scope questions; more official pages for the Python standard library.
4. Human review of a sample of live answers, semantic flags and the UI; push and let CI build the new image.
