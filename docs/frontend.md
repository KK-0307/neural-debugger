# The Neural Debugger application (Milestone 6)

A developer-facing web app (`frontend/`, Next.js 16 + React 19 + Tailwind 4) on top of `POST /api/v1/debug` and the new
`POST /api/v1/feedback`. It does what the brief asked: paste a stack trace or question, submit, see the diagnosis and suggested
fixes, inspect the source citations, understand the uncertainty, and give feedback.

## What a developer sees

| Area | Behaviour |
|---|---|
| **Input** | Large monospace box, character counter (3 to 20,000), documentation filter (auto / Python 3.12 / PyTorch 2.14 / FastAPI 0.162), three example prompts, Ctrl/Cmd+Enter to submit, Clear, Cancel while waiting, an elapsed-time indicator (free-tier models can take 10-30 s) and a plain statement that the text is sent to the server and its LLM provider. |
| **Answer** | A status banner (Answered / Partly answered / Not enough documentation / Error) with a confidence badge; **Likely cause**; numbered **Suggested fixes**; **Example code** with a copy button, labelled *illustrative, never run*; documentation **conflicts**; **what the documentation did not cover**; **keep in mind** caveats (long lists collapse; duplicates of the coverage message are hidden). |
| **Citations** | Every statement carries chips (`E1`, `E2`...). Clicking one scrolls to that source, highlights it and opens **"Text the model was shown"** so the claim can be checked against the exact passage. Sources are split into *cited* and *also retrieved*, each with framework and version and an external link to the official page (http(s) only, `rel="noopener noreferrer"`). |
| **Uncertainty** | "How sure is this?" restates what the validators measured, nothing more: documentation coverage %, statements removed (no citation / unsupported / contradicted by a number or signature), statements **flagged** (the cited text may say the opposite; a name or number is not in it), partly supported statements, discarded citations, version mismatches, conflicts. Per-statement badges: *Check this*, *Not in cited text*, *Partly supported*. A truthful provenance line says whether a real model, the offline test provider, or nobody wrote the text, and that no human reviewed it. |
| **Abstentions** | Shown as a deliberate outcome: why, what was missing, and the closest documentation as search leads, instead of an empty or invented answer. |
| **Errors** | Specific, actionable messages: no LLM configured on the server (with the variables to set and the retrieved sources still listed), server busy, model unavailable / rate limited, timeout, request budget used, validation error, API unreachable (with the command to start it). Retry where retrying makes sense. |
| **Feedback** | Helpful / Not helpful, reason tags, an optional comment (<= 1,000 chars). **The question text is stored only if the user ticks an opt-in box** (off by default). Server side the comment is secret-redacted and written to an append-only `var/feedback/feedback.jsonl` (git-ignored, capped at 20 MB). |
| **History** | The last 10 answers are kept **only in this browser** (`localStorage`), reopen instantly without a new request, and can be cleared. Storage failures never break the app. |
| **Quality of life** | Dark and light themes follow the system; responsive from a 375 px phone (verified: no horizontal overflow) to desktop with a sticky sources column; keyboard accessible, labelled landmarks, `aria-live` status, focus moves to the result. |

## Safety and privacy decisions

* Model output is rendered as **inert text** by React; there is no `dangerouslySetInnerHTML`. A test feeds `<img onerror>` and
  `<script>` and `javascript:` URLs and asserts nothing executes or links.
* The backend already strips non-documentation URLs and markup; the UI defends again (`safeHref`).
* The privacy notice sits next to the submit button; the question is never stored by feedback unless opted in; history never leaves the browser.
* No authentication, rate limiting per user, accounts or analytics exist. The server-side request cap (`ND_LLM_MAX_REQUESTS`),
  pacing and concurrency limit are unchanged.

## Backend additions

* `POST /api/v1/feedback` (201 `{id, stored_question}`; 422 on invalid input; 503 `feedback_storage_full`).
* `POST /api/v1/debug` now also returns `evidence_gate {level, coverage}`, per-source `excerpt` (the text shown to the model),
  per-claim `semantic` / `semantic_reason`, and the new validation counters; all additive.
* Settings: `ND_FEEDBACK_DIR`, `ND_RAG_PROFILE`, `ND_CORPUS_SUPPLEMENT_DIRS`, `ND_EMBEDDING_DIR` (see `.env.example`).

## Run it

```bash
make dev-backend            # needs ND_LLM_PROVIDER=gemini + GEMINI_API_KEY in .env for real answers (free tier)
make dev-frontend           # http://localhost:3000   (NEXT_PUBLIC_API_URL defaults to http://localhost:8000)
# no key at all? the offline test provider assembles answers from documentation sentences (stilted, but exercises the UI):
ND_LLM_PROVIDER=extractive make dev-backend
```

## Verification actually performed

* `npm test`: 38 tests (8 older + 30 new): citation parsing and URL safety, uncertainty explanations, history, every answer
  variant (answered, partial with flags, abstained, errors), the full submit -> answer -> feedback flow against a real local HTTP
  server, cancel, retry, history reopen, the XSS case and the feedback privacy defaults. Lint, `tsc`, and `next build` pass.
* **In a real browser** (the Claude Code built-in browser, Chromium-based) against the real backend: example prompt -> answer
  with citation chips and sources; chip click opens the source text; an out-of-scope question -> abstention view; feedback sent
  and found on disk with `question: null`; reload restores history; 375 px layout. A live run with the real Gemini free-tier
  model through the UI answered the "coroutine was never awaited" example in 3.7 s with medium confidence, 100% coverage and
  two "partly supported" badges.
* Backend: 9 new feedback/response-shape tests (privacy defaults, secret redaction, validation, full store, response fields).

## Not done / not verified

* No human has used or reviewed it; accessibility was designed in and spot-checked in one browser, **not audited** (no screen
  reader, no axe run); only Chromium at two widths; no Safari/Firefox.
* No streaming (one request, one answer), no auth, no persistence of answers server-side, no export/share, no database-backed
  feedback (a JSONL file), no analytics. Feedback is stored but nothing consumes it yet.
* The answer quality is whatever the pipeline delivers: on independently written questions it answered 39% of in-scope ones
  (`docs/rag-coverage.md`); the UI shows that honestly rather than hiding it.

## Demo build (Milestone 5.3)

`NEXT_PUBLIC_DEMO_MODE=static npm run build` exports a static site (`frontend/out/`) whose home page renders `DemoApp` instead of
`DebuggerApp`: a picker over 12 recorded answers (`lib/demo-data.json`, built by `scripts/build_demo_data.py`), the same `AnswerView`
in `readOnly` mode (feedback form replaced by a note) and no network access. Details, limits and the deployment checklist:
[demo.md](demo.md).
