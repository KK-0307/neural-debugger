# Neural Debugger

An AI debugging assistant for developers: paste an error or stack trace and get a grounded
explanation, troubleshooting steps, and citations to indexed documentation. Retrieval combines BM25
with a **custom transformer encoder written from scratch on PyTorch tensors**, evaluated against
baselines, inside a RAG pipeline.

> **Project status: early development (Milestones 1-3 of 8; the original brief's custom-transformer
> milestone is still ahead).** Built and verified: API skeleton, a reproducible documentation corpus,
> a labeled evaluation set, BM25, and now **pretrained dense / hybrid / reranked retrieval with measured
> results**. Not built yet: the custom-trained transformer, RAG generation, the debugging UI.
> [PROGRESS.md](PROGRESS.md) lists exactly what is done and verified. No result is claimed that has
> not been measured.

## What exists today

* **Corpus** `nd-corpus-v1`: 207 pages / 4,154 sections / 4,995 chunks from Python 3.12, PyTorch 2.14
  and FastAPI 0.162.0 docs, pinned and fetched politely; every chunk has source URL (with anchor),
  framework, version and hash; all 4,154 citation anchors verified on the published sites.
  [docs/data-pipeline.md](docs/data-pipeline.md)
* **Evaluation set** `nd-eval-v1`: 203 queries with deterministic, leakage-checked splits. **Labels are
  LLM-authored and not human-reviewed**; judgments were pooled from BM25 and later expanded with neural
  candidates. [docs/evaluation.md](docs/evaluation.md)
* **Retrieval systems** (`configs/neural.yaml`): BM25; dense (`gte-small`, 67 MB, MIT) over persistent
  versioned embeddings with an exact vector index; hybrid BM25 + dense via reciprocal-rank fusion; an
  optional cross-encoder reranker; framework filtering; CPU/MPS/CUDA with CPU fallback.
  [docs/neural-retrieval.md](docs/neural-retrieval.md)

**Measured results** (NDCG@10, authored queries; full tables, intervals and caveats in the docs):

| | dev (train+val, n=138) | test (n=32), original labels | test, expanded labels |
|---|---:|---:|---:|
| BM25 (frozen baseline) | 0.552 | 0.597 | 0.569 |
| Dense (gte-small) | 0.666 | 0.580 | 0.604 |
| Hybrid (tuned RRF) | 0.680 | 0.601 | 0.626 |
| Hybrid + cross-encoder | 0.685 | 0.653 | 0.666 |

On the 138 dev queries dense/hybrid beat BM25 clearly (paired difference up to +0.13, 95% CI
[+0.09, +0.17]). On the small held-out test set the direction agrees but **almost nothing is
statistically significant** (intervals about +/-0.1); the dev gain partly reflects selection optimism.
The cross-encoder did not help significantly on dev and costs 10x the latency. Judgments are
incomplete (up to 65% of neural top-10 results were never judged under the original labels), which is
why the table shows two judgment sets. Read the limitations before quoting anything.

Try it on the real documentation:

```bash
uv sync --extra data --extra neural          # PyTorch + sentence-transformers (BM25 needs only `data`)
make demo Q="RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x784 and 128x10)"          # BM25
make demo-neural Q="RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x784 and 128x10)"   # hybrid
```

For that query BM25's top hits are Python's generic `RuntimeError` / `BrokenExecutor` pages, while the
hybrid returns PyTorch's *matmul* and *broadcasting* sections (first neural query is slower: model load).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), Node 24+, (PostgreSQL 16 only for the `/health` DB
check and later milestones).

```bash
cp .env.example .env     # adjust if needed
make install             # uv sync --extra data + npm ci
make check               # lint + type check + all tests (what CI runs, minus Docker builds)
make demo Q="KeyError"   # BM25 search over the real corpus
make eval-bm25           # re-score the frozen BM25 baseline on train+val (test split is guarded)
make freeze-verify       # fail if the frozen baseline / neural configuration drifted
make dev-backend         # http://localhost:8000/health  (needs a database, see make dev-db)
make dev-frontend        # http://localhost:3000
```

Rebuild the corpus from the network (cached after the first run; ~24 MB, 1 request/s):
`make corpus && make verify-urls && make dataset`.

## Layout

| Path | Contents |
|---|---|
| `src/neural_debugger/backend` | FastAPI app |
| `src/neural_debugger/data` | ingestion, chunking, dedup, labeled dataset, splits, leakage checks |
| `src/neural_debugger/ml` | retrieval (tokenizer, BM25, encoders, embeddings, vector index, RRF, reranker, router), evaluation (metrics, paired stats, judgments, freeze); later custom transformer + training |
| `tests/` | pytest suites (`backend`, `data`, `ml`) |
| `frontend/` | Next.js app |
| `data/` | `corpus/`, `embeddings/gte-small/` (committed), `eval/` (labels, pooled judgments, results), `manifests/`, `NOTICE.md` (licences) |
| `configs/`, `scripts/`, `docker/`, `docs/` | sources config, authoring/rendering scripts, Dockerfile, documentation |

Docs: [architecture](docs/architecture.md) · [data pipeline](docs/data-pipeline.md) ·
[evaluation](docs/evaluation.md) · [neural retrieval](docs/neural-retrieval.md) ·
[technical decisions](docs/technical-decisions.md).

macOS note: if `import neural_debugger` fails inside `.venv`, run `chflags -R nohidden .venv` (see
the neural-retrieval doc); `pytest` and `make` do not depend on it.
Environment variables are documented in [.env.example](.env.example). Never commit credentials.
