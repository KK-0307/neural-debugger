# Progress

**Current milestone:** 2 - Documentation retrieval: **complete and verified, with caveats listed below.**
Stopped for review before Milestone 3 (custom PyTorch transformer), as requested.

| # | Milestone | Status |
|---|---|---|
| 1 | Project foundation | Done (Docker builds and CI runs not yet verified, see Known issues) |
| 2 | Documentation retrieval | **Done** (labels are LLM-authored and unreviewed by a human) |
| 3 | Custom PyTorch transformer | Not started |
| 4 | Training and evaluation | Not started |
| 5 | RAG integration | Not started |
| 6 | Full-stack application | Not started |
| 7 | Performance and reliability | Not started |
| 8 | Public beta and documentation | Not started |

## Baseline freeze and label review (before Milestone 3)

- 102 judgments over 38 dev queries re-read by the same assistant (agent second pass, **not** independent, **not** human): 71 confirmed, 23 marginal, 6 uncertain, 2 weak; 6 queries flagged (`data/eval/label_review.yaml`). Labels were not edited. Test-split labels not reviewed.
- `data/eval/baseline_freeze.json` + `make freeze-verify` + a test pin corpus, queries, splits, BM25 config and results.
- Test split protected: scoring it needs `--final-test` and is logged (`protocol.py`, `test_access_log.jsonl`).

## Package-layout migration (requested before M2) - done

Replaced the symlink layout with a conventional `src/neural_debugger/{backend,data,ml}` package
(`hatchling`, editable install), tests in `tests/{backend,data,ml}`, Dockerfile in `docker/`.
Verified: no symlinks remain in the repo; a built wheel installed in a clean venv serves `/health`
from an unrelated directory (200 with PostgreSQL 16 up, 503 once it is stopped); the Dockerfile's
two-step `uv sync` was replayed outside Docker; ruff, strict mypy, pytest and the frontend
lint/typecheck/test/build all pass. Rationale: `docs/technical-decisions.md`.

`httpx` is kept. The Starlette `TestClient` warning was investigated and is documented in
`docs/technical-decisions.md`; pytest now treats **all** warnings as errors except that one exact
message/category/module.

## Milestone 2 - Definition of Done

| Criterion | Result |
|---|---|
| Real documentation ingested | Yes: 207 pages (Python 3.12, PyTorch 2.14, FastAPI 0.162.0 @ `7a0a3f0b`), 4,154 sections, 4,995 chunks, 521k words; 97 requests, ~24 MB decompressed |
| Documents searchable with BM25 | Yes: `make demo Q="..."`, 0.7 s index build, ~0.75 ms mean query |
| Results carry valid source URLs + metadata | Yes: every chunk has doc id, title, URL+anchor, framework, version, heading path, content hash, ingestion time. **207/207 pages resolve and 4,154/4,154 anchors exist** on the published sites (`data/manifests/url_verification.json`) |
| Labeled held-out evaluation set | Yes: 203 queries, train 122 / val 39 / test 42 (authored-only 103 / 35 / 32) |
| Metrics calculated correctly | Hand-computed unit tests + cross-check against scikit-learn `ndcg_score` on 60 random cases + property tests |
| Dataset leakage checks pass | Yes: no near-duplicate/identical query across splits, splits equal the deterministic rule, no empty split, all labels reference real sections |
| M1 functionality intact | Yes (see migration section) |
| Tests, lint, types pass | Yes (below) |
| README / PROGRESS updated | Yes |
| Actual baseline results recorded | Yes: `data/eval/results/*.json`, `docs/evaluation.md` |

### Measured BM25 results (test split, authored queries only, n=32; 95% bootstrap CI)

| | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 (default) | 0.329 [0.21, 0.45] | 0.567 [0.42, 0.71] | 0.692 [0.56, 0.82] | 0.666 [0.52, 0.80] | 0.597 [0.47, 0.72] |
| random floor (seed 0) | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 |

Including the official-FAQ slice (n=42): R@1 0.465, R@10 0.765, NDCG@10 0.684 - inflated, because
those queries are the indexed headings. By framework (all splits, authored+FAQ): Python NDCG@10
0.71, PyTorch 0.64, FastAPI 0.56. Full tables, slices, latency and failure analysis:
[docs/evaluation.md](docs/evaluation.md). No other retriever (pretrained, custom, hybrid) has been
measured; none is claimed.

Notable finding: the code-aware tokenizer is **not shown** to beat plain word tokens
(dev paired difference +0.005 NDCG@10, 95% CI [-0.027, +0.018]).

## Verification actually performed

| Command | Result |
|---|---|
| `uv run pytest` | **92 passed** (backend 12, data 47, ml 33) |
| `uv run ruff check .` / `ruff format --check .` | clean (75 files) |
| `uv run mypy -p neural_debugger` / `mypy tests` (strict) | no issues (44 / 21 files) |
| `npm run lint`, `typecheck`, `npm test`, `npm run build` (frontend) | clean; 8 tests passed; production build ok |
| `python -m neural_debugger.data.ingest build` (twice) | first run 97 requests; re-run 0 requests, byte-identical corpus |
| `python -m neural_debugger.data.ingest verify-urls` | pages 207/207, anchors 4,154/4,154 |
| `python -m neural_debugger.data.dataset build` | OK (integrity + leakage) |
| `python -m neural_debugger.ml.evaluation.bm25_ablation` | dev only (train+val); test untouched |
| `python -m neural_debugger.ml.evaluation.evaluate --retriever bm25 / --framework-filter / random` | results in `data/eval/results/` |
| `python -m neural_debugger.ml.retrieval.demo` | real queries return ranked sections with URLs |
| Built wheel in clean venv + real PostgreSQL 16 | `/health` 200, then 503 after stopping Postgres |

Bugs found and fixed by the new tests during M2: anchorless API definitions produced duplicate
section/chunk ids (33 colliding chunks); code-block language detection and Markdown table cell
formatting in the parsers; all-numeric tokens (`3.12`) were split; the pipeline now fails loudly on
duplicate section ids.

## Known issues / not verified

- **Labels are not human-verified** (`human_reviewed_queries: 0`). 170 of 203 queries and all pooled
  additions were written/judged by the assistant; only the 33 FAQ queries have independent
  provenance (and are lexically trivial). Pooling used BM25 only, so BM25 is favoured and later
  systems must be re-pooled before comparison. Test n is 32-42 (wide intervals).
- **Docker is not installed on this machine**: `docker build` / `docker compose up` still have not
  been run (their steps were replayed outside Docker). **CI has never run** (no GitHub remote).
- Local PostgreSQL has no pgvector (deferred to M4 as instructed); no database schema exists yet
  (deviation from the original M2 bullet; see technical-decisions).
- Published pages change over time; the committed corpus + raw cache are the reproducibility anchor.
- The corpus is small (46 PyTorch pages); real-user queries do not exist yet.
- `next/font/google` needs network at build time.
- Windows is untested.

## Remaining work

Milestones 3-8. Next: **Milestone 3** - transformer encoder from scratch (embeddings, scaled
dot-product + multi-head attention with explicit masks, layer norm, FFN, residuals, pooling,
projection, L2 norm), dual encoder, InfoNCE loss, correctness tests (attention vs a reference
implementation, masking, gradients, checkpointing, determinism). Before comparing neural systems
fairly, plan: human review of a sample of labels and pooling with several systems.
