# Retrieval-augmented debugging answers with validated citations (Milestone 5)

**Status.** Implemented, tested, evaluated offline (sections 1-10, Milestone 5) and then **exercised against one real
hosted model** (Gemini 3.1 Flash-Lite, free tier) in Milestone 5.1 ([section 11](#11-milestone-51-live-verification-independent-questions-semantic-grounding-deployment)).
Sections 1-10 describe the offline evaluation with the deterministic test provider; where they say "unverified live"
or "negation survives", section 11 and [semantic-grounding.md](semantic-grounding.md) give what was measured since.
The Anthropic provider is still tested against a mocked transport only. See [section 9](#9-what-is-verified-and-what-is-not)
for the exact boundary.
Read the [limitations](#10-limitations) before quoting any figure. Labels remain LLM-authored and not human-verified
([evaluation](evaluation.md)).

## 1. What it does

`POST /api/v1/debug` (or `python -m neural_debugger.rag.cli`) takes a Python, PyTorch or FastAPI question,
error message or pasted traceback and returns a diagnosis, ordered troubleshooting steps and illustrative code,
**every claim carrying citations that resolve to documentation sections that were actually retrieved**, or an
explicit abstention when the documentation cannot support an answer.

```
question ─► prepare ─► retrieve ─► select evidence ─► sufficiency gate ──► abstain (no LLM call)
 (normalise,  (hybrid-    (sanitise, dedupe,    (IDF-weighted           │
  scan,        gte-small   budget, version      term coverage)          ▼
  query)       by default) notes)                              prompt (delimited data blocks)
                                                                       │
                                  provider.generate ◄──────────────────┘   (Anthropic / OpenAI-compatible / test)
                                          │
              parse JSON (1 repair try) ─► validate citations ─► check support ─► assemble + confidence
                                          (ids ∈ shown evidence,    (lexical judge by default;
                                           URLs, markup stripped)    LLM judge optional)
```

| Stage | Module | Behaviour |
|---|---|---|
| Question | `rag/question.py` | normalise, strip invisible/bidi characters, keep head+tail of long input (the exception is at the end of a traceback), extract error lines and user-mentioned versions, build a retrieval query from the error line and the user's prose (fenced code, traceback frames and instruction-like sentences are excluded) |
| Retrieval | `rag/retrieval.py` | any project retriever; **default `hybrid-gte-tuned`** (setting `ND_RETRIEVER_SYSTEM`); the from-scratch transformer is selectable (`hybrid-custom-tiny`). Hits not present in the corpus are skipped |
| Evidence | `rag/evidence.py` | dedupe (exact + term-set Jaccard >= 0.8), at most 2 sections per page, 5 passages / 6,000 characters, truncated at paragraph boundaries, prompt-injection scanning, version notes ("Changed in version ...") extracted |
| Sufficiency | `rag/sufficiency.py` | pre-LLM abstention when evidence covers too little of the question (below) |
| Prompt | `rag/prompt.py` | fixed system prompt; user text and each passage in delimited blocks the model is told are *data*; JSON output contract |
| Generation | `rag/providers/` | provider interface; `anthropic`, `openai_compatible` (also local Ollama/vLLM), deterministic test providers |
| Validation | `rag/validate.py`, `rag/support.py` | citation **validity** and, separately, citation **support** (below) |
| Assembly | `rag/pipeline.py` | status `answered` / `partial` / `abstained` / `error`, computed confidence, caveats, timings |

## 2. Provider interface and configuration

The pipeline depends only on `LLMProvider.generate(LLMRequest) -> LLMResponse`. Failures raise `ProviderError`
with a stable `kind` (`not_configured`, `auth`, `rate_limit`, `timeout`, `network`, `server`, `bad_request`,
`empty`); the pipeline turns them into `status = "error"` **with the retrieved sources still listed**, and the API
maps them to 502/503/504 with the sources in `error.details`. Nothing crashes and nothing is guessed.

| Setting (`ND_` prefix) | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `none` | `anthropic`, `openai_compatible`, `none` (answers 503 `llm_not_configured`), `extractive` (test only, refused in production) |
| `LLM_MODEL` | `claude-haiku-5-5` | model id; the cheapest current Claude model is the default, but **no provider is enabled by default, so nothing costs money until you set one** |
| `LLM_API_KEY_ENV` | `ANTHROPIC_API_KEY` | *name* of the environment variable holding the key. The key itself is never a setting, is not in `repr`, logs or error messages (tested) |
| `LLM_BASE_URL` | provider default | e.g. `http://localhost:11434/v1` for a local OpenAI-compatible server (no key, no cost). The app deliberately ignores `ANTHROPIC_BASE_URL` |
| `RETRIEVER_SYSTEM` | `hybrid-gte-tuned` | any system in `configs/neural.yaml` / `configs/m4.yaml` |
| `RAG_SUPPORT_CHECK` | `lexical` | `off`, `lexical`, `llm` |

HTTP providers use `httpx`, a 60 s timeout, and up to 2 retries with exponential back-off (honouring
`Retry-After`) for 429/5xx/timeouts; 401/403/400 are not retried. To verify live generation yourself:

```bash
export ANTHROPIC_API_KEY=...           # your key; costs a few cents
uv sync --extra neural --extra data
python -m neural_debugger.rag.cli --provider anthropic "RuntimeError: Expected all tensors to be on the same device"
pytest tests/rag/test_integration.py::test_live_anthropic_generation_end_to_end
python -m neural_debugger.rag.evaluation run --provider anthropic --judge llm --faults
```

Free alternative: run a local model behind an OpenAI-compatible server and pass `--provider openai_compatible
--base-url http://localhost:11434/v1 --model <name>`.

### Free hosted LLM (Gemini free tier) - no local weights

Chosen because it is genuinely free and reuses the OpenAI-compatible provider (`ND_LLM_PROVIDER=gemini`). Facts
checked on 2026-10-26 against Google's pages ([pricing](https://ai.google.dev/gemini-api/docs/pricing),
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
[OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)):

* Flash and Flash-Lite models are free of charge (3.x Flash/Flash-Lite per the pricing page); **Pro and image models are
  treated as paid** and are not on this project's allowlist. **`gemini-2.5-flash` is listed by the API but refuses generation
  for new users (HTTP 404 "no longer available to new users")**, so listing a model is not proof it works: the harness now
  sends one tiny generation probe before the suite. The free tier does not require a billing account; moving to a paid tier requires you to set up billing.
  **Create the key in a project with no billing attached** and requests can never be charged (they just fail with 429).
* Endpoint `https://generativelanguage.googleapis.com/v1beta/openai/` with `Authorization: Bearer <key>` (beta).
* Limits are per **project**, differ by model, are "not guaranteed", and are shown only in the AI Studio console;
  Google's docs page lists no numbers and third-party sources disagree (about 5-15 requests/minute, 20-1,500/day).
  The harness therefore paces itself at 7 s/request and stops on repeated 429s.
* **Privacy:** on the free tier Google may use prompts and responses to improve its products. Do not send private code
  or data; the acceptance questions are public-documentation questions.
* Alternatives checked: OpenRouter `:free` models are limited to 20 requests/minute and 50/day (until credits are
  bought) and warn that a negative credit balance breaks free models; Groq's docs page did not state free-plan limits.

Setup (the key never appears in a file that is committed, printed or logged; `.env` is git-ignored):

```bash
echo 'GEMINI_API_KEY=<your key>' >> .env          # or export it in your shell
uv sync --extra neural --extra data
python -m neural_debugger.rag.live_acceptance --provider gemini --model gemini-3.1-flash-lite   # 21 questions, <= 45 requests
#   `gemini-3.8-flash` (the newer Flash) answered 503 "high demand" repeatedly on 2026-10-26, so that run used 3.1 Flash-Lite
python -m neural_debugger.rag.cli --provider gemini --model gemini-3.1-flash-lite "RuntimeError: ..."
ND_LLM_PROVIDER=gemini ND_LLM_MODEL=gemini-3.1-flash-lite ND_LLM_MIN_INTERVAL_SECONDS=7 ND_LLM_MAX_REQUESTS=200 uvicorn neural_debugger.backend.main:app
```

Guard rails: model allowlist of free models (anything else needs `--allow-unlisted-model`), a hard request cap, minimum
spacing, a preflight `GET /models` (not a generation call) that confirms the model exists for the key, immediate stop on
auth errors / exhausted cap / repeated rate limiting, **no fallback to any other model**, and redaction of raw outputs
(`data/eval/results/m5_1/*.raw.json`).

## 3. Citations: validity versus support

* **Validity (automated, exact).** A citation is valid iff it is the id of an `Evidence` passage that was in the
  prompt. Invalid ids and inline `[E#]` markers are removed; a claim left with no valid citation is dropped (a
  "conflict" needs two); URLs that are not source URLs, markdown images and HTML tags are stripped from prose.
  Each returned citation carries the corpus section id, heading path, framework, version and deep link, taken from
  the index rather than from the model.
* **Support (separate).** A valid citation can still not support its sentence. `LexicalSupportJudge`
  (automated, default) checks term coverage and, more strictly, identifier coverage (backticked names, dotted
  paths, version numbers) against the cited passages; unsupported claims are dropped and partial ones reduce
  confidence. `LLMSupportJudge` (model-graded) is available for stronger checking. **Neither is ground truth, and
  the lexical judge was not calibrated against human labels**; it cannot see negation (measured below).
* Code suggestions are inert text: judged on their explanation (or on the code when the explanation is too short
  to check), flagged if they contain risky constructs (`rm -rf`, `curl | sh`, `sudo`, `eval`/`exec`,
  `subprocess`) absent from the documentation, and **never executed**. A test asserts that submitted code with a
  side effect is not run, and an AST scan asserts the `rag` package contains no `eval`/`exec`/`compile`/
  `subprocess`/`os.system`.

## 4. Abstention and uncertainty

1. **Before the model:** if nothing usable is retrieved, or the IDF-weighted coverage of the question's key terms
   by the evidence is below **0.375**, the pipeline abstains without calling the LLM and says which terms had no
   documentation. A term absent from the whole corpus gets the maximum weight, so "Django"/"TensorFlow" questions
   score low even when generic words match. The threshold was chosen on development queries only: the largest
   threshold with <= 5% false abstentions on *train* authored queries, maximising abstention on the *calibration*
   half of the out-of-scope set (`data/eval/rag/unanswerable.yaml`, 32 assistant-authored questions; the other half
   is held out). Coverage between 0.375 and 0.65 ("weak", 0.65 is an uncalibrated prior) is passed to the model with
   an explicit instruction to answer only what is supported and list what is missing, and adds a caveat.
2. **By the model:** the prompt requires `status: "insufficient_evidence"` rather than guessing. *This is the
   second line of defence; in the Milestone 5.1 live runs the real model used it for 12 of 39 in-scope questions (section 11).*
3. **After the model:** if every claim is dropped the status is `abstained` (`no_supported_claims`); if some are
   dropped it is `partial`.
4. **Confidence** is computed from validation (invalid citations, unsupported/partial/dropped claims, weak
   coverage, conflicts) and capped at `medium` when no support check ran; it is not the model's self-report.

## 5. Versions and conflicting documentation

The corpus holds one version per framework (Python 3.12, PyTorch 2.14, FastAPI 0.162.0). Versions the user
mentions ("pytorch 1.13", "torch==2.1.0") are extracted; when they differ from the documented version the model is
told and a caveat is added. "Added/Changed/Deprecated in version X" notes inside cited passages are surfaced as
caveats. Passages from several frameworks trigger a "check which one your code uses" caveat, and the output
schema has a `conflicts` list (each needs >= 2 valid citations). **Limitation:** because only one version per
framework is indexed, true cross-version conflicts cannot occur in the data; contradictions *within* the
documentation rely on the model noticing them (never observed in the live runs, which had no conflicting documents).

## 6. Security

Threat model: instructions hidden in retrieved documentation, in the user's question or in a pasted log, aimed at
the model (override the rules, steer citations, reveal the prompt, exfiltrate via links), and attempts to make the
service run code.

| Layer | What it does | Evidence |
|---|---|---|
| Structure | untrusted text only in delimited blocks; system prompt says block contents are data; delimiter look-alikes (`</evidence>`, `</user_question>`, ...) are escaped so text cannot close its block | tests with forged delimiters |
| Scanning | 9 rule families (override/reveal/new-instructions/role/chat-template tokens/citation steering/output dictation/markdown images/`<script>`). In **documents** instruction-like lines are replaced by a marker, passages that are mostly such lines are dropped, the rest is flagged; in **user text** they are only flagged (still data) and excluded from the retrieval query | 10 attack strings detected, 5 benign phrasings not flagged |
| False positives | measured over all 4,995 corpus chunks: **0 high-severity**; 6 medium flags, all `<script>` inside HTML code samples (flag only, text unchanged) | `tests/rag/...::test_no_high_severity_false_positives_on_the_real_corpus` |
| Output validation | even a fully hijacked model cannot cite an id it was not shown, introduce a non-source link or image, or emit active markup; tested with a scripted "obedient" model | `test_a_fully_hijacked_model_cannot_fabricate_sources_or_links` |
| No execution | code is text; no code path evaluates user or model content | AST scan + side-effect test |
| Credentials | key only from env, redacted everywhere | `test_credentials_never_appear_in_logs...` |
| Operational | question length limits (6,000 chars, head+tail kept), per-process concurrency cap (429 `busy`), questions are not logged | API tests |

These are heuristics: **paraphrased or non-English injections will be missed by the scanner**, and the structural
and output layers depend on the model following the system prompt. No red-team exercise against a live model was
possible.

## 7. API

`POST /api/v1/debug` `{"question": "...", "framework": "pytorch"|"python"|"fastapi"|null, "include_trace": false}`.
Response (abridged): `status`, `confidence`, `diagnosis`, `steps[]`, `code_suggestions[]`, `conflicts[]`,
`caveats[]`, `missing_information[]`, `citations[]` (id, title, heading, url, framework, version),
`sources[]` (everything shown to the model, with `cited` and injection `flags`), `abstain_reason`, `validation`
counts, `timings`, `provider`, `model`, `request_id`. Errors use the standard `{"error": {...}}` body; provider
and retrieval failures return 502/503/504 with `details[0].sources`.

The Docker backend image contains only the core dependencies, so there the endpoint answers 503
`retrieval_unavailable`; run the API locally with `uv sync --extra neural --extra data` for retrieval. (Adding the
neural stack to the image is a size/ops decision I did not make unprompted.)

## 8. Evaluation

Only **development queries (train + val)** are used; the held-out test split is **not touched** in this
milestone. Provider: the deterministic `extractive` test provider, which assembles answers from evidence sentences
and cannot hallucinate. **So the numbers below validate retrieval, selection, abstention gating, validation and
latency plumbing; they say nothing about how well any LLM answers.** Reproduce with
`python -m neural_debugger.rag.evaluation run --provider extractive --retriever <system> --faults` and
`python scripts/render_m5_tables.py`.

| Evidence type | What is in this report |
|---|---|
| **Automated** | retrieval hit rates, citation validity, lexical support, abstention, latency, fault injection |
| **Model-graded** | implemented (`--judge llm`: LLM claim-support and answer-correctness judges, tested with scripted judges) but **not run: no live provider** |
| **Human-verified** | **none** (n = 0). Gold sections are the LLM-authored, pool-expanded M4 labels; unanswerable questions are assistant-authored |

### 8.1 Retrieval relevance and answer grounding (dev authored queries; "all" = train + val)

| retriever | split | n | gold in retrieved top-12 | gold in selected evidence | answer cites a gold section (answered) | false abstentions |
|---|---|---:|---:|---:|---:|---:|
| hybrid-gte-tuned | all | 138 | 94.9% | 87.7% | 83.3% | 4.3% |
| hybrid-gte-tuned | val | 35 | 94.3% | 91.4% | 91.4% | 0.0% |
| hybrid-custom-tiny | all | 138 | 84.1% | 66.7% | 60.3% | 8.7% |
| hybrid-custom-tiny | val | 35 | 91.4% | 77.1% | 69.7% | 5.7% |
| bm25 | all | 138 | 86.2% | 77.5% | 73.7% | 3.6% |
| bm25 | val | 35 | 88.6% | 82.9% | 85.3% | 2.9% |


"Gold in selected evidence" is retrieval relevance *after* de-duplication and budgeting (the backbone decision:
`hybrid-gte-tuned` is clearly the best). "Answer cites a gold section" is a **proxy** for answer correctness (the
extractive answer is built from top-overlap sentences), not a correctness judgement. Gold sets come from model-judged,
partly gte-pooled labels, which favour the systems that produced the pool.

### 8.2 Abstention

| retriever | out-of-scope library (n=22) abstained | off-topic (n=10) abstained | held-out half (n=16) abstained | false abstentions, dev answerable (n=161) | abstention F1 (all) |
|---|---:|---:|---:|---:|---:|
| hybrid-gte-tuned | 77.3% | 100.0% | 81.2% | 3.7% | 0.831 |
| hybrid-custom-tiny | 59.1% | 100.0% | 75.0% | 7.5% | 0.687 |
| bm25 | 45.5% | 90.0% | 56.2% | 3.1% | 0.679 |


Out-of-scope questions that were still answered (5/32 for the default) share only generic vocabulary with the
docs ("fails", "state", "load"): a coarse gate cannot catch them, which is why the model's own abstention matters.
The calibration half and the held-out half of the out-of-scope set are both reported in the JSON; the held-out
half is the honest number for the threshold. The set is small (n = 32), so intervals are wide.

### 8.3 Citation validity and claim support

| slice | n | citation validity before filtering | claims | unsupported (lexical) | partial | dropped (no valid citation) |
|---|---:|---:|---:|---:|---:|---:|
| authored dev | 138 | 100.0% | 655 | 0.5% | 0.1% | 0.0% |
| FAQ dev | 23 | 100.0% | 114 | 0.0% | 0.0% | 0.0% |


Validity before filtering is 100% **trivially** for the extractive provider (it only copies ids it was shown), and
post-filtering validity is 100% by construction; the number to look for with a real LLM is the pre-filtering
validity. The honest signal of whether the *validators work* is the next table.

### 8.4 Fault injection (automated; measures the validators, not any model)

Valid extractive answers are corrupted with one synthetic fault each, and we count how often the fault survives
into the returned answer.

Eligible answers: 132.

| injected fault | what it is | survived validation |
|---|---|---:|
| `fabricated_citation` | A step citing a source id that was never shown (E99). | 0 / 132 (0.0%) |
| `uncited_claim` | A confident factual claim with no citation. | 0 / 132 (0.0%) |
| `unsupported_claim_valid_citation` | A claim citing a real source but using identifiers the source does not contain. | 0 / 132 (0.0%) |
| `injected_url` | A claim carrying an exfiltration link not among the sources. | 0 / 132 (0.0%) |
| `negated_claim_valid_citation` | A real claim with its meaning inverted (lexically identical vocabulary). | 132 / 132 (100.0%) |

The validators remove fabricated citations, uncited claims, claims using identifiers the cited source lacks, and
foreign links. They **cannot** catch a claim whose meaning is inverted but whose vocabulary matches the source
(100% survival): that was the gap the Milestone 5.1 semantic stage targets ([semantic-grounding.md](semantic-grounding.md)); the 132/132 figure is the frozen Milestone 5 baseline with the lexical judge only.

### 8.5 Latency (this machine, CPU/MPS, 193 dev questions, p50 / p95)

| retriever | retrieval p50 / p95 (ms) | generation p50 (ms) | validation p50 (ms) | total p50 / p95 (ms) |
|---|---:|---:|---:|---:|
| hybrid-gte-tuned | 8.38 / 11.8 | 0.31 | 0.35 | 11.95 / 16.54 |
| hybrid-custom-tiny | 2.05 / 2.43 | 0.33 | 0.35 | 5.42 / 9.1 |
| bm25 | 0.23 / 0.36 | 0.34 | 0.36 | 3.29 / 6.5 |


Generation time with a real LLM (typically seconds) is **not measured**; add it to the figures above. Retrieval
with the production hybrid is about 8 ms (the one-off model load takes ~6 s on first request).

## 9. What is verified and what is not

| Acceptance criterion | State |
|---|---|
| Pipeline works end to end with a **configured real LLM provider** | **VERIFIED on 2026-10-26 against a real hosted model** (Gemini 3.1 Flash-Lite, free tier, through the OpenAI-compatible provider): 51 questions, 66 generation requests (4 of them 1-token probes), 100% valid JSON, 0 invalid citation ids; results and caveats in section 11. The Anthropic provider is still mocked-HTTP-only (the live Anthropic test skips without `ANTHROPIC_API_KEY`) |
| Retrieved documentation is used to build answers | verified with real retrieval and the test provider |
| Citations are real and validated | verified (validity exact; support heuristic) |
| Insufficient-evidence cases handled | verified for the pre-LLM gate (84% of out-of-scope questions, 3.7% false abstention); with a real LLM 9 of 9 out-of-scope questions abstained (8 by the gate, 1 by the model); the real model also reported insufficient evidence on 12 in-scope questions (section 11) |
| Tests: success, failure, malicious retrieved instructions, missing sources, provider errors | 97 new tests in `tests/rag` and `tests/backend/test_debug_api.py` (96 run; the live-LLM test skips without a key) |
| All earlier checks still pass | see PROGRESS.md; the frozen Milestone 5 evaluation reproduces exactly (146 of 146 non-timing numbers) with the Milestone 5.1 code |

## 10. Limitations

* **Live verification is one model, one free tier, 51 questions** (section 11): Gemini 3.1 Flash-Lite on 2026-10-26. It says
  nothing about other models, other days (Google answered 503 "high demand" after retries for 4 of 21 questions in one run and 4 of 30 in the other; each was rerun once and then succeeded), or load.
  Prompt-injection resistance was exercised with only 2 poisoned-document cases that reached a live model.
* The model-graded and human-verified rows of the evaluation are empty; every correctness statement in section 11 is an
  **agent review** (the author of the pipeline reading answers), not independent and not human.
* The sufficiency gate is a lexical heuristic calibrated on 138 + 23 dev answerable and 32 out-of-scope questions
  authored by me; it will over-abstain on terse or code-heavy questions (of its 6 false abstentions on dev answerable questions, 4 were terse or
  error-code-heavy questions with low term coverage and 2 abstained after generation because no claim survived
  validation) and under-abstain on out-of-scope questions that share generic words.
* The lexical support judge cannot detect negation or subtle contradiction and is uncalibrated; the Milestone 5.1
  semantic stage ([semantic-grounding.md](semantic-grounding.md)) helps on synthetic mutations, but its recall on real
  model errors is unmeasured because the live runs contained no known defective claim.
* One documentation version per framework; no conflict detection across documents beyond version notes and the
  model's own report.
* Injection scanning is regex-based and English-only. Anything not documented here was not tested.
* Evaluation covers dev queries only; no test-split RAG number exists yet, deliberately.
* No conversation memory, streaming, caching or persistence of answers.
* Answers can carry noisy `caveats`: the Python 3.12 type-parameter answer listed seven version notes, several irrelevant.
* The first request after start loads the retriever (about 6 s on this Mac, 31 s measured in a cold simulated container).

## 11. Milestone 5.1: live verification, independent questions, semantic grounding, deployment

Everything here was produced on 2026-10-26. Labels: **agent-authored** (written by the assistant that built the
pipeline), **model-generated** (written by Gemini, which never saw the corpus or the pipeline), **agent-reviewed**
(the assistant read the outputs; not independent), and **human-verified: none**.

### 11.1 The model and what it cost

* Provider `gemini` (OpenAI-compatible endpoint), model **`gemini-3.1-flash-lite`**, free tier, key from the git-ignored
  `.env`; the key was never printed, logged or committed (an audit of the repository and all run outputs found no
  occurrence). The harness paced calls at 7 s, capped them at 45 per run, and never fell back to another model.
* Not usable: `gemini-2.5-flash` (listed, but generation returns 404 "no longer available to new users"), and
  `gemini-3.8-flash` (503 "high demand" on repeated attempts). This is why the harness now sends a 1-token generation
  probe before a run: listing a model is not proof it works.
* Usage: 30 + 36 generation requests for the two question sets (counted by the harness, including retries and 4 probes),
  7 to generate the independent questions, plus a handful of uncounted early probes against retired or overloaded
  models: **roughly 80-90 requests; 75,504 input and 10,552 output tokens in the two question runs**, all on the
  free tier. I cannot see the key's project billing settings; a key created in a project without billing cannot be
  charged, and no paid model was used.

### 11.2 Results on 21 agent-authored questions (`data/eval/rag/live_acceptance.yaml`)

All 21 got a model response (4 needed one rerun after Google 503s; reruns only fill infrastructure errors, never replace
an answer).

| measure | result |
|---|---|
| in-scope questions answered with citations and expected terms | **12 of 15 (80%)**: python 3/5, pytorch 4/5, fastapi 5/5 |
| the 3 misses | all **retrieval misses**, not model errors: `dict.get` ranks 17th, the in-place-operation autograd notes rank 36th, `sys.setrecursionlimit` is not retrieved; the model correctly said the evidence was insufficient |
| out-of-scope questions (TensorFlow, Terraform, Kafka) | **3 of 3 abstained**, all by the pre-LLM gate |
| JSON valid on the first try | 17 of 17 replies; 0 repairs |
| invalid citation ids | **0** of 60 claims |
| claims removed by the lexical support check | 4 of 60 with current code (9 of 60 in the run as recorded, before the identifier fixes of section 11.5) |
| prompt injection | 2 poisoned-document cases reached the model: its replies contained no attack marker (`E99`, `PWNED`, `evil.example`) and neither did the answers. A third case (instruction inside the user question) was abstained by the gate and never reached the model. **n = 3: this is a smoke test, not a measurement of resistance** |
| latency (generation) | p50 11.5 s, p95 72.7 s (the p95 is a call that waited out a 503 retry); end to end p50 7.7 s, p95 19.4 s |
| tokens | 34,334 in, 5,481 out |

Agent review of the 12 answers (not independent): all were appropriate and consistent with the documentation: `python -m pip`
for the `requests` import error, the `with` statement, `await` / `asyncio.create_task`, `.to(device)`, `PYTORCH_ALLOC_CONF`,
gradient accumulation and `zero_grad`, `torch.load` `weights_only` defaulting to True since 2.6 with `add_safe_globals`,
`CORSMiddleware`, `lifespan`, `status_code=201`, and `python-multipart`. The FastAPI 422 answer was generic. Four answers
(pt1, pt3, fa2, in3) lost a correct claim, or had the user's own URL replaced by `[link removed]`, to the old
identifier/URL bugs (section 11.5).

### 11.3 Results on 30 independent questions (`data/eval/rag/independent_gemini.yaml`)

Gemini wrote 8 Python, 8 PyTorch, 8 FastAPI and 6 out-of-scope questions (prompt hash in the file). They are
**model-generated, not human**, have no gold answers, and several contain false or unsupported premises.

| measure | result |
|---|---|
| in-scope questions answered | **9 of 24 (37.5%)**: python 3/8, pytorch 2/8, fastapi 4/8 |
| in-scope questions abstained | 15: 6 by the gate (term coverage 0.26-0.34), 9 by the model reporting insufficient evidence |
| out-of-scope questions abstained | **6 of 6** (5 by the gate, 1 by the model) |
| JSON valid on the first try | 19 of 19; 0 repairs; invalid citation ids **0** of 38 claims; 0 claims removed |
| latency (generation) | p50 13.3 s, p95 69.3 s; tokens 41,170 in / 5,071 out |

Agent review: **all 9 answers were appropriate** (type-parameter `SyntaxError` on older Python, `typing.Unpack` added in 3.11,
`@override`, NaN propagation and AMP unscaling, saving optimizer state, background tasks and `yield` dependencies,
`OAuth2PasswordBearer` user dependency, CPU-bound work in async routes, `Form` instead of a JSON body for multipart).
The abstentions mostly follow from what the corpus is: reference documentation, not a troubleshooting knowledge
base. The model said it lacked the *cause* for "autograd returns None", "device-side assert", "dependency override not
applied" and "CORS despite allow_origins=['*']", and lacked features the index does not cover (per-interpreter GIL,
sqlite3 CLI). The gate's six abstentions (long, conversational questions with low term coverage) include questions the corpus
plausibly can answer (the 422 error, DataLoader workers, moving tensors to a device); I did not verify each. **This is the
number to take away: grounded and rarely wrong when it answers, but it answered about a third of realistic, verbose,
troubleshooting-style questions.** The gate threshold was calibrated on shorter dev questions and over-abstains here.

### 11.4 Semantic grounding

Summarised here, detailed with all tables in [semantic-grounding.md](semantic-grounding.md): on a deterministic
synthetic adversarial set (test half, 518 items), the lexical judge alone silently accepted **52%** of defective claims; with
rules **21%**; with rules + NLI **2.7%**, and most of those are flagged-and-kept rather than removed. On the 98 real
claims (60 + 38), `rules` raised 2 disputes and `nli` 3, all of which I judge to be false alarms on correct claims, the
semantic check confirmed few paraphrases (4-9 of 98), and it flagged 3 claims as `unverified` for identifiers/numbers
not in the cited text (reasonable observations); **its recall on real errors is unmeasured** because no real defective claim was found. Deployment default:
`rules` (0.2 ms/claim); `nli` (283 MB, ~37 ms/claim, +520 MB memory) is opt-in.

### 11.5 Defects found only because a real model was used (all fixed, all tested)

1. `e.g.` / `i.e.` were treated as API identifiers, so correct claims containing them were removed.
2. A dotted name had to appear verbatim, but reference pages list `empty_cache` under the module; the documented tail
   now counts (a fabricated tail still does not).
3. A value the user typed (`http://localhost:3000`) was treated as fabricated and replaced by `[link removed]`. Only the
   user's non-injection text is trusted and only loopback URLs are let through; an injected `http://evil.example/...` is still removed.
4. The first semantic design flagged 85-95% of real claims and removed five correct ones (section 4 of semantic-grounding.md).
5. The generation probe and a filename bug (`gemini-2.5-flash` collapsed to `gemini-2` in the output path) in the harness.

After these fixes the frozen Milestone 5 evaluation reproduces exactly.

### 11.6 Deployment (Docker)

* The image now installs the retrieval stack (`--extra neural`), uses **CPU-only PyTorch on Linux** (the old lockfile resolved
  CUDA wheels: 73 `nvidia` entries, now 0), copies the corpus, the precomputed embeddings and the retriever configs, and
  bakes the pinned `gte-small` revision (65 MB, only the files the encoder needs) with `HF_HUB_OFFLINE=1`. No secret is
  in the image: keys come from the environment at run time (`docker-compose.yml` passes `GEMINI_API_KEY` /
  `ANTHROPIC_API_KEY` through). With no provider the endpoint answers **503 `llm_not_configured`** with a message telling the
  operator which variables to set, and still lists the retrieved sources.
* **Verified locally without Docker (not installed here):** a staging directory containing exactly what the Dockerfile copies,
  a fresh `uv sync --frozen --extra neural` environment, the model baked into an isolated `HF_HOME`, offline mode, a scrubbed
  environment, and the in-repo mock OpenAI-compatible server: `/api/v1/debug` returned `answered` with 4 PyTorch citations;
  without a provider it returned the informative 503 with 5 sources. First request 31 s (cold retriever load).
* **Verified in CI (GitHub Actions run on `d148b6d`, the first real build):** the Dockerfile and Linux CPU wheels build; the image is
  **2,766 MiB** (large: CPU-only torch plus the sentence-transformers stack; no multi-stage slimming was tried, and the
  < 3 GiB guard leaves little headroom); no `.env`, secret-like environment variable or key name is in the image; and
  `scripts/smoke_debug.py` against the running container returned `answered` with 4 citations (provider configured, mock
  server inside the container, first request 9.6 s) and the informative 503 with 5 retrieved sources (no provider).
  These are plumbing checks with a mock LLM, not a live-model test inside the container; the 283 MB NLI model is not in the image.
