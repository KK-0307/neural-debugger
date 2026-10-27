# Public read-only demo

**Status: prepared and tested locally; not deployed.** Nothing has been published, no API key has been placed anywhere except the
git-ignored `.env` on my machine, and nothing will be deployed without your explicit go-ahead (checklist in section 5).

There are two parts. The first is what I recommend publishing; the second exists for completeness and is **not** recommended for
a first public release.

## 1. The static recorded demo (recommended)

A static website with **no backend and no API key**. It shows 12 real answers that the system produced with Gemini 3.1 Flash-Lite
(free tier) on 2026-10-27, chosen by me from development recordings and three canonical prompts, never from the frozen held-out or
final-evaluation sets. The examples include behaviour the project is supposed to be honest about: a headline PyTorch answer with
a statement removed as unsupported and low confidence, a generic FastAPI answer, an answer whose wrong-framework citation was
removed, an over-abstention that is a known miss, and a correct refusal for an out-of-scope question.

What a visitor can do: pick an example, read the answer, open citation chips to see the text the model was shown, follow source
links to the official documentation. What they cannot do: type a question, send anything anywhere, or leave feedback (the
feedback form is replaced by a note). The banner says so on every page: "Recorded demo: read-only, no server behind it".

How the examples are made trustworthy:

* They are not typed in by hand. `scripts/build_demo_data.py` replays each *recorded raw model reply* through the **current**
  retrieval, evidence selection, gate, citation validation and semantic check, serialises it exactly as `POST /api/v1/debug`
  would, and **refuses to build** if the replay does not reproduce the recorded status and cited sections. No model call happens
  at build time.
* The curated list and my notes are in `data/demo/spec.yaml`; the recordings are in `data/eval/results/m5_3/demo_recording/` (one
  transient Google 503 on one question was re-run, only that one). The generated `frontend/lib/demo-data.json` and the recordings are
  covered by the M5.3 freeze manifest.
* The notes ("judged correct", "plausible") are my own agent judgement, not a human review; the banner and page do not claim
  otherwise.

Build and preview locally:

```bash
python scripts/build_demo_data.py                                  # only if spec.yaml or the pipeline changed
cd frontend && NEXT_PUBLIC_DEMO_MODE=static npm run build          # writes frontend/out/ (static export)
python3 -m http.server 3100 --directory out                        # or: preview config "demo-static" in .claude/launch.json
```

`NEXT_PUBLIC_DEMO_MODE=static` switches `next.config.ts` to `output: "export"` and turns off two experimental flags that the static
export does not support (`cacheComponents`, `partialPrefetching`); a normal build is unchanged (`output: "standalone"`).

What I checked in a real browser against the exported site with no backend running: after clicking an example the page made no
network request, there is no text area and no feedback button, the "feedback is disabled" note is shown, source links point to
`docs.python.org` / `docs.pytorch.org` / `fastapi.tiangolo.com`, and the layout works at 375 px. Automated: `frontend/tests/DemoApp.test.tsx`.
Where to host the folder: any static host (GitHub Pages, Netlify, Cloudflare Pages). Because the site is only files, there is
nothing to attack and nothing to pay for.

**What it does not show:** a visitor cannot try their own error. The page says how to run the project locally for that.

## 2. Read-only API mode (optional; do not publish a live endpoint yet)

`ND_DEMO_MODE=readonly` makes the backend safe(r) to expose, for a live "try it" later. It does **not** make a public Gemini-backed
endpoint a good idea on a free key (section 4).

| protection | behaviour |
|---|---|
| feedback | `POST /api/v1/feedback` returns 403 `demo_readonly`; nothing is stored |
| question size | over `ND_DEMO_MAX_QUESTION_CHARS` (2000) returns 422 `demo_question_too_long` |
| per-client limits | `ND_DEMO_RATE_PER_MINUTE` (4) and `ND_DEMO_RATE_PER_DAY` (40) sliding windows; 429 with `Retry-After` |
| global budget | `ND_DEMO_BUDGET_PER_DAY` (150) answers per day across all visitors; then 429 `demo_budget_exhausted` until the oldest hit ages out |
| tracing | `include_trace` is ignored (no internals in public responses) |
| logging | the request log line never contains question text, and in this mode no client address |
| client identity | the connecting address; `X-Forwarded-For` is honoured **only** if `ND_TRUST_PROXY_HEADERS=true` (set it only behind a proxy you control, otherwise anyone can spoof it) |
| memory | the per-client table is bounded (10,000 clients, least recently seen evicted) |

Existing protections still apply: a hard cap on provider requests per process (`ND_LLM_MAX_REQUESTS`), pacing, and bounded concurrency
(`ND_RAG_MAX_CONCURRENT`). Tests: `tests/backend/test_demo_mode.py` (per-minute / per-day / global limits, bounded memory,
feedback 403 with nothing stored, size cap and hidden trace, `Retry-After`, `X-Forwarded-For` only when trusted, default mode unaffected).

Limits of this mode, stated plainly: the limiter is in memory and per process (a restart resets it; two replicas double the
budget); the regular UI (`DebuggerApp`) has not been adapted to it (its feedback form would receive a 403); I have not
load-tested it; and a determined attacker with many addresses can still spend the daily budget, which is the point of having a global
budget as the backstop.

## 3. The key

* The static demo contains no key and needs none. The built `frontend/out/` and `demo-data.json` were scanned for key-shaped strings
  and for the real key value: none.
* A live endpoint would hold `GEMINI_API_KEY` as a host environment variable (never in the image, repo, build args or frontend
  bundle; `NEXT_PUBLIC_*` variables are shipped to browsers and must never hold it).
* **Free-tier data use.** Google's free-tier terms allow prompts and responses to be used to improve their products. A public
  endpoint would therefore forward visitors' pasted errors (which can contain paths, hostnames or tokens) to Google under
  that arrangement. The recorded demo avoids this entirely; a live demo would need a visible notice, and a paid key (which I will not
  introduce) would be the cleaner answer.

## 4. Why I recommend against a live endpoint for now

The free tier allows roughly 10 requests per minute and a small daily quota shared by all visitors; one burst of traffic, or one
person scripting the endpoint, ends the demo for everyone for the day, and Gemini's "high demand" 503s already interrupted several of
my own evaluation runs (re-run files `rerun_infra_errors/`). Answer quality is also visibly unfinished (38% of in-scope questions answered; `docs/rag-reliability.md`),
so an open text box invites a first-time visitor to hit a refusal. The recorded demo shows honest behaviour on chosen questions;
that is a fair portfolio presentation if it is labelled as recorded, which it is.

## 5. Deployment checklist (nothing below has been done)

1. You approve the plan, and which host(s).
2. Static site: `NEXT_PUBLIC_DEMO_MODE=static npm run build`, upload `frontend/out/`. No key, no environment variables.
3. Only if you also want a live endpoint: provision the host secret yourself (or tell me where); set `ND_DEMO_MODE=readonly`,
   `ND_TRUST_PROXY_HEADERS=true` only behind your proxy, `ND_CORS_ORIGINS=["https://<your-demo-site>"]`,
   `ND_LLM_MAX_REQUESTS` to a number you are comfortable with, `ND_RAG_WARM_ON_STARTUP=true`; put the "free-tier data use" notice in
   the UI; check the first request takes ~10 s cold (Docker smoke test: 10.3 s).
4. After deploying: run `python scripts/smoke_debug.py <url>`, confirm the response headers and that `/api/v1/feedback` returns 403,
   then try the burst limit once yourself.
5. Rotate the key afterwards if it was ever pasted into a place other than the host secret store.
