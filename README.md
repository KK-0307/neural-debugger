# Neural Debugger

An AI debugging assistant for developers: paste an error or stack trace and get a grounded
explanation, troubleshooting steps, and citations to indexed documentation. Retrieval combines BM25
with a **custom transformer encoder written from scratch on PyTorch tensors**, evaluated against
baselines, inside a RAG pipeline.

> **Project status: early development (Milestones 1-2 of 8 complete).** Built and verified: the
> API skeleton (`/health`), a status page, a reproducible documentation corpus, a labeled evaluation
> set, and a **BM25 baseline with measured results**. Not built yet: the custom transformer,
> dense/hybrid retrieval, RAG, the debugging UI. [PROGRESS.md](PROGRESS.md) lists exactly what is
> done and verified. No result is claimed that has not been measured.

## What exists today (Milestone 2)

* **Corpus** `nd-corpus-v1`: 207 pages / 4,154 sections / 4,995 chunks from Python 3.12, PyTorch
  2.14 and FastAPI 0.162.0 docs, fetched politely (robots-aware, cached) and pinned. Every chunk
  carries doc id, title, URL (with anchor), framework, version, section heading, content hash and
  ingestion time. All 4,154 citation anchors were verified on the published sites.
  [docs/data-pipeline.md](docs/data-pipeline.md)
* **Labeled dataset** `nd-eval-v1`: 203 queries (error messages, questions, official FAQ headings)
  with deterministic, leakage-checked train/val/test splits. **Labels are LLM-authored and not yet
  human-reviewed**; only the 33 FAQ queries have independent provenance.
  [docs/evaluation.md](docs/evaluation.md)
* **BM25 baseline** (own implementation, code-aware tokenizer) and metrics (Recall@1/5/10, MRR,
  NDCG@10, bootstrap CIs). Measured on the held-out test split, authored queries only (n=32):
  Recall@1 0.33, Recall@10 0.69, MRR 0.67, NDCG@10 0.60 (intervals of roughly +/-0.13; see the doc
  for caveats - pooled labels favour BM25, test set is small).

Try it on the real documentation:

```bash
uv sync --extra data
make demo Q="RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu"
```

```
Query: RuntimeError: Expected all tensors to be on the same device, ...
3 sections in 0.4 ms (BM25 over 4995 chunks, corpus 3d3ae6ff3ac8)

  1. [pytorch 2.14] Tensor Attributes > torch.device   (score 30.22)
     https://docs.pytorch.org/docs/2.14/tensor_attributes.html#torch-device
  2. [pytorch 2.14] CUDA semantics   (score 29.61)
     https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-semantics
  3. [pytorch 2.14] torch.load > torch.load   (score 27.77)
     https://docs.pytorch.org/docs/2.14/generated/torch.load.html#torch.load
```

(That third hit is not a good answer: BM25 matches tokens, not meaning. That gap is what the
neural retriever is for.)

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), Node 24+, (PostgreSQL 16 only for the `/health` DB
check and later milestones).

```bash
cp .env.example .env     # adjust if needed
make install             # uv sync --extra data + npm ci
make check               # lint + type check + all tests (what CI runs, minus Docker builds)
make demo Q="KeyError"   # BM25 search over the real corpus
make eval-bm25           # reproduce the baseline numbers (data/eval/results/)
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
| `src/neural_debugger/ml` | retrieval (tokenizer, BM25), evaluation (metrics, CLI); later transformer + training |
| `tests/` | pytest suites (`backend`, `data`, `ml`) |
| `frontend/` | Next.js app |
| `data/` | `corpus/` (committed), `eval/` (labels + results), `manifests/`, `NOTICE.md` (licences) |
| `configs/`, `scripts/`, `docker/`, `docs/` | sources config, authoring/rendering scripts, Dockerfile, documentation |

Docs: [architecture](docs/architecture.md) · [data pipeline](docs/data-pipeline.md) ·
[evaluation](docs/evaluation.md) · [technical decisions](docs/technical-decisions.md).
Environment variables are documented in [.env.example](.env.example). Never commit credentials.
