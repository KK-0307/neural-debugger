# Deployment: the whole application on a free host

The live application is **one container**: the FastAPI service (retrieval, validated RAG, Gemini calls) also serves the real
Next.js frontend, exported to static files, from the same origin (`docker/fullstack.Dockerfile`). Nothing is a recorded or mocked
demo: every answer on the deployed site is produced live by the retrieval stack and Gemini 3.1 Flash-Lite (free tier).

Live: **https://neural-debugger.onrender.com** (Render free web service, deployed by the owner from `render.yaml`). The sections
"What was verified" and "Live URL" say exactly what was checked against the public URL and what was not.

## Why a single container, and why Render

Hard requirement: no billing, no paid plan. What I found (checked 2026-10-27, from each provider's documentation and search; terms
change, re-check before relying on them):

| host | free tier | verdict |
|---|---|---|
| **Render** (free web service, Docker) | 512 MB RAM, <1 vCPU, 750 instance hours/month, sleeps after 15 min idle (~1 min wake-up), documented without a payment method | **chosen**: the only no-card option the container fits |
| Hugging Face Spaces (Docker) | 16 GB RAM, 2 vCPU, but the Hub documentation says Docker/Gradio Spaces now require a paid plan (PRO) to create; only Static Spaces are free | not free for this app |
| Google Cloud Run, Fly.io | free allowances need a billing account / card | excluded by "no billing" |
| Oracle Cloud Always Free | ~1 GB micro VMs or Arm VMs, but card verification and VM administration | excluded (card) |
| Koyeb, SnapDeploy free tiers | 512 MB or less, sleeping, limited hours | same class as Render, less documented |
| Static hosts (Cloudflare Pages, Netlify, GitHub Pages) | free, but cannot run Python / PyTorch | would force a recorded demo, which is not the goal |

The binding constraint is **memory**: the retrieval stack (CPU PyTorch + `gte-small` + the corpus) is the large part. Measured in CI
(`docker run --memory=512m --memory-swap=512m`, real image, real retrieval, mock LLM): the container served the site, config,
privacy checks, a real retrieval answer and a burst of 8 requests with `OOMKilled=false`; cgroup anonymous memory ended at
**468 MB** of the 512 MB limit (of which ~25-30 MB was the mock LLM process that does not exist in production) plus 46 MB of
reclaimable file cache. That is a **thin margin (~70 MB in production)**, so concurrency is capped at 2 answers
(`ND_RAG_MAX_CONCURRENT`), BLAS threads at 1 and allocator arenas at 2. If Render's free instance ever OOM-kills the service,
the logs will say so; the fix would be a paid plan or a smaller retrieval stack, neither of which I enable.

## What runs

| piece | setting |
|---|---|
| image | `docker/fullstack.Dockerfile` (Node stage builds the export with `NEXT_PUBLIC_API_URL=""`, `NEXT_PUBLIC_EXPORT=1`; Python stage = backend image + `/app/site`) |
| service | `render.yaml` (Blueprint): free plan, Docker, `autoDeploy: false`, health check `/health` |
| LLM | Gemini 3.1 Flash-Lite via the OpenAI-compatible endpoint; key only as an environment variable on the host |
| mode | `ND_DEMO_MODE=readonly`: no feedback, 2,000-character cap, no trace, 3 questions/minute and 20/day per visitor, 120 answers/day for everybody, hard cap of 400 Gemini requests per process, 7 s spacing |
| privacy | request log lines carry no question text and no client address; validation errors never echo input; the UI tells visitors before they type that their text goes to Google's free tier, which may use it to improve its products |
| state | none persisted: the filesystem is ephemeral, history lives in the visitor's browser |

Environment variables are listed in `render.yaml` and `.env.example`; the only secret is `GEMINI_API_KEY`.

## Deploying (what the owner does; I do not hold any account)

1. **GitHub**: the repository is public (`https://github.com/KK-0307/neural-debugger`), so Render needs no GitHub permissions beyond reading it.
2. **Render**: sign up with GitHub (no payment method is required for a free web service; **do not add one**, so no overage can
   ever be billed: without a payment method Render suspends free services instead).
3. One click: <https://render.com/deploy?repo=https://github.com/KK-0307/neural-debugger> (or Dashboard -> **New -> Blueprint** -> pick this
   repository). Render reads `render.yaml`.
4. When asked for `GEMINI_API_KEY`, paste your free-tier key (Google AI Studio). It is stored as an encrypted environment variable.
5. Apply. The first build takes several minutes (it installs CPU PyTorch and builds the frontend). Afterwards:

```bash
python scripts/smoke_public.py https://<your-service>.onrender.com --answers 3 --check-limits
```

That script checks, from outside: the frontend is served, security headers, the config the UI reads, feedback refused, an
over-long question rejected, malformed input rejected without echoing it, an out-of-scope question declined before the model with no
trace, real Python / PyTorch / FastAPI answers with valid HTTPS citations of the right framework, and the rate limit (using
out-of-scope questions, so it spends no Gemini quota).

## What was verified

| check | where | result |
|---|---|---|
| image builds; no `.env` or credential-like variable inside | CI docker job | pass |
| container inside a 512 MB limit: site, config, privacy, rate limit, retrieval answer | CI docker job, `scripts/smoke_public.py` | pass, `OOMKilled=false`, anon 468 MB |
| full stack with **real Gemini** on a local machine (exported frontend served by FastAPI, read-only mode) | `scripts/smoke_public.py --answers 3 --check-limits` | Python answered (medium confidence, 3 citations), PyTorch partial (low confidence, 2 citations), limits enforced |
| UI in a real browser (desktop and 375 px), notice, answer, abstention | screenshots in `docs/assets/` | as shown in the README |
| **public URL, from outside** (`scripts/smoke_public.py https://neural-debugger.onrender.com --answers 3 --check-limits`) | 2026-10-27 | pass: frontend + security headers, 404 page, config with free-tier notice, feedback refused (403), over-long and malformed input rejected without echo, out-of-scope question declined before the model with no trace, real answers for Python (answered, medium, 3 citations, 5.6 s), PyTorch (partial, low, 2 citations, 6.3 s) and FastAPI (answered, medium, 5 citations, 7.0 s), rate limit returns 429 with Retry-After |
| public URL in a phone-size browser (390 px, touch) | 2026-10-27 | no horizontal overflow, deployment notice shown, an example answered, feedback hidden with the read-only note, only the site's own origin contacted (no third-party requests) |
| fresh start | 2026-10-27 | the service was asleep: the first request returned the page after 45 s (Render's wake-up, includes loading the retriever); the following requests were fast |

## Cost and free-tier limits

* **Cost: $0.** No payment method is attached anywhere; Render's free plan, Google AI Studio's free tier, GitHub Actions for a
  public repository.
* The site **sleeps after 15 minutes idle**; the next visitor waits about a minute (longer if the free CPU is slow to load the
  retriever: the first request after wake-up waits for the model to load, in the background from process start).
* **Shared free Gemini quota**: when Google rate-limits or says "high demand", visitors get a plain-language error, not a crash.
  Per-visitor, per-day and per-process caps stop one visitor from using it all.
* 750 free instance hours a month; free services are suspended for the rest of the month if exhausted.
* Free-tier prompts may be used by Google to improve its products: that is why the UI warns before anything is sent.

## Live URL

**https://neural-debugger.onrender.com**: verified from outside on 2026-10-27 as in the table above.

Not verified on the public site, and why: **Gemini error handling** (a 503 "high demand" or exhausted quota cannot be forced on
demand; the mapping to plain-language errors is covered by the backend and frontend tests and was seen during development),
**the per-day and global budgets** (exercising them would spend the shared quota; they are unit-tested), **memory over a long
session** (CI shows 468 MB of 512 MB at steady state with the mock model; if Render ever kills the process, its logs say so), and
behaviour after the 750 free hours are used up (the service is suspended until the next month).
