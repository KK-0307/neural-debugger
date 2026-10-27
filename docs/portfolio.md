# Portfolio notes: demo walkthrough, resume bullets, evidence map

Everything here is for presenting the project. Each number is traceable to a file in the repository (last section). All
evaluation labels are LLM-authored and nothing was human-reviewed; say so if asked, it is a strength of the write-up, not a
weakness.

## 1. A 60-90 second walkthrough (about 190 words, 75 s at a normal pace)

Screen: the app, desktop width, light theme. Have the PyTorch example and one out-of-scope question ready.

| time | on screen | say |
|---|---|---|
| 0:00 | landing page | "This is Neural Debugger. You paste a Python, PyTorch or FastAPI error, and it answers only from the official documentation, with a citation on every statement." |
| 0:10 | click **PyTorch: tensors on different devices**, **Diagnose** | "It retrieves with BM25 plus dense embeddings, fused; then a gate decides whether the evidence is good enough to ask the model at all." |
| 0:25 | answer appears; hover a citation chip, open *Text the model was shown* | "Every citation resolves to a section the model actually saw, and I can read that text. Statements the documentation only partly supports carry a badge, and one was removed entirely." |
| 0:40 | scroll to **How sure is this?** | "Confidence is not the model's opinion. It is built from validator counters: how much of the question the docs cover and how many statements were only partly backed." |
| 0:50 | type the Kubernetes question, **Diagnose** | "Now something the docs don't cover." |
| 0:55 | abstention view | "It declines, before calling the model, and shows the closest pages. On a fresh 54-question set it declined 12 of 12 out-of-scope questions." |
| 1:05 | switch to the README results table | "I measured everything: hybrid retrieval beat BM25 on dev, my from-scratch transformer lost and I report that, and the latest improvement, 28 to 38 percent of questions answered, is not statistically significant. The README says so." |
| 1:15 | end on the repo/CI badge | "FastAPI, Next.js, Docker, 335 backend tests, CI that smoke-tests the container." |

Recording tips: use the committed screenshots as a fallback if the free-tier model is slow (answers can take 10-30 s; cut the
wait in editing, do not speed up the screen so it looks instant). If the model returns a 503 "high demand", retry; it is a
free-tier behaviour the UI handles with a plain-language error.

## 2. Resume bullets (three, quantified, each defensible in an interview)

**Backend / software engineering**
- Built and shipped a full-stack debugging assistant (FastAPI, Next.js, Docker) that answers Python, PyTorch and FastAPI errors from
  4,995 documentation chunks with machine-validated citations: 0 invalid citation ids across 67 answered claims on a fresh
  54-question evaluation, 335 backend and 47 frontend tests, strict typing, and a CI pipeline that builds the images and
  smoke-tests the container inside a 512 MB memory limit.

**Backend / production safeguards**
- Engineered the safeguards for a public service on a free LLM tier: a pre-model abstention gate (12 of 12 out-of-scope questions
  declined, most before any model call), per-client and global rate limits with bounded memory, a read-only mode, request caps,
  and privacy-by-default logging (no question text, no echoed input), verified from outside the container by a standalone
  smoke-test script.

**AI / ML engineering**
- Benchmarked BM25, pretrained dense, hybrid and reranked retrieval plus a transformer encoder implemented from scratch in PyTorch:
  hybrid RRF raised NDCG@10 from 0.552 to 0.680 on 138 dev queries (paired 95% CI +0.09 to +0.17) while the from-scratch model
  lost (test NDCG@10 0.31 vs 0.56 for BM25 on the same labels), reported as a negative result; then diagnosed an evidence gate that wrongly refused 18 of 36
  answerable verbose questions and fixed it with a dense-similarity signal (1 of 36), lifting the in-scope answer rate from 28.6% to
  38.1% on a frozen fresh set without loosening out-of-scope refusals, and documenting that the gain is not statistically
  significant (n = 42, p = 0.125).

If you need them shorter, keep the first sentence of each and the headline number. Do not drop "evaluation labels are
model-written": interviewers ask.

## 3. Claims and the evidence behind them

| claim | evidence |
|---|---|
| 4,995 chunks, 207 pages (Python 3.12, PyTorch 2.14, FastAPI 0.162) | `data/manifests/corpus_manifest.json`, [data pipeline](data-pipeline.md) |
| Hybrid NDCG@10 0.680 vs BM25 0.552 (dev, n=138); CI +0.09 to +0.17; 79 better / 23 worse | [neural retrieval](neural-retrieval.md), `data/eval/results/m3/dev_final/`, `make freeze-verify` (M3) |
| From-scratch transformer test NDCG@10 0.314 dense, 0.531 hybrid; BM25 0.560 on the same (M4) labels | [ml-model.md](ml-model.md), `data/eval/results/m4/` |
| 12/12 out-of-scope declined; 12/42 → 16/42 answered; 0 invalid ids in 67 claims; quality 11/4/1 | `data/eval/results/m5_3/final_summary.json`, raw replies in `final_before/` and `final_after/`, [rag-reliability.md](rag-reliability.md) |
| Gate: 18 → 1 of 36 verbose answerable refusals | `data/eval/results/m5_3/gate_diagnosis_before.json`, `gate_diagnosis_after.json`, `gate_regression.json` |
| p = 0.125 (not significant) | exact sign test on 4 gained / 0 lost, `final_summary.json` |
| 335 Python tests, 47 frontend tests | CI run logs (GitHub Actions), `make check` |
| Container fits a 512 MB limit | CI step "Full-stack container inside a 512 MB limit" (OOMKilled false); step summary of the run |
| Free-tier latency p50 7.4 s / p95 21 s generation | `final_summary.json` (`latency_ms.after`), the free tier varies |
| Evaluation sets and results unchanged since recorded | `data/eval/*_freeze.json`, `make freeze-verify` |

Claims deliberately **not** made: that answers are correct in general (no human review), that the system beats a general
chatbot (never compared), that recall of the semantic claim checks on real model errors is known (unmeasured), or any
statistically significant end-to-end improvement.
