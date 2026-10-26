# Robust to the macOS hidden-.pth problem (see docs/technical-decisions.md): never rely on the
# editable-install .pth alone.
export PYTHONPATH := $(CURDIR)/src

.PHONY: install lint format typecheck test test-backend test-frontend check \
        dev-db dev-backend dev-frontend up down \
        corpus verify-urls dataset eval-bm25 eval-ablation freeze-verify demo \
        demo-neural embed eval-dev

install:
	uv sync --extra data
	npm --prefix frontend ci

lint:
	uv run ruff check .
	uv run ruff format --check .
	npm --prefix frontend run lint

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy -p neural_debugger
	uv run mypy tests
	npm --prefix frontend run typecheck

test-backend:
	uv run pytest

test-frontend:
	npm --prefix frontend test

test: test-backend test-frontend

# --- Milestone 2: corpus, labeled dataset, BM25 evaluation -------------------------------------
corpus:  ## fetch (cached, polite) + parse + chunk + dedupe the pinned documentation corpus
	uv run python -m neural_debugger.data.ingest build

verify-urls:  ## check every cited page and anchor resolves on the published site
	uv run python -m neural_debugger.data.ingest verify-urls

dataset:  ## merge YAML label sources -> queries.jsonl, run integrity + leakage checks
	uv run python -m neural_debugger.data.dataset build

eval-bm25:  ## re-score the frozen BM25 baseline on train+val (test is protected; see protocol.py)
	uv run python -m neural_debugger.ml.evaluation.evaluate --retriever bm25

freeze-verify:  ## fail if the frozen baseline (M2) or neural configuration (M3) drifted
	uv run python -m neural_debugger.ml.evaluation.freeze verify
	uv run python -m neural_debugger.ml.evaluation.freeze verify-m3
	uv run python -m neural_debugger.ml.evaluation.freeze verify-m4
	uv run python -m neural_debugger.ml.evaluation.freeze verify-m5

embed:  ## (re)generate chunk embeddings: make embed M=gte-small
	uv run python -m neural_debugger.ml.retrieval.embed --model $(M)

eval-dev:  ## dev (train+val) evaluation of one system on expanded judgments: make eval-dev S=hybrid-gte-tuned
	uv run python -m neural_debugger.ml.evaluation.system_eval --system $(S) --qrels expanded

demo-neural:  ## neural/hybrid search over the real docs: make demo-neural Q="RuntimeError: ..."
	uv run python -m neural_debugger.ml.retrieval.demo --system hybrid-gte-tuned $(Q)

eval-ablation:  ## dev-only (train+val) BM25 configuration study
	uv run python -m neural_debugger.ml.evaluation.bm25_ablation

demo:  ## BM25 search over the real docs: make demo Q="KeyError"
	uv run python -m neural_debugger.ml.retrieval.demo $(Q)

# Everything CI runs (minus Docker builds).
check: lint typecheck test

dev-db:
	docker compose up -d db

dev-backend:
	uv run uvicorn neural_debugger.backend.main:app --reload --port 8000

dev-frontend:
	npm --prefix frontend run dev

up:
	docker compose up --build

down:
	docker compose down
