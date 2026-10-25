# Technical decisions

Each entry: decision, why, and what we gave up.

## M1 - Foundation

**One root `pyproject.toml`, `uv`, Python 3.12, standard `src/` layout.** *(Revised after review; the
first version used symlinks.)* The backend, data pipeline and ML code are subpackages of a single
installable package, `src/neural_debugger/{backend,data,ml}`, built with `hatchling` and installed
editable by `uv sync`. Tests live in `tests/{backend,data,ml}`. One environment and lockfile
(`uv.lock`) means versions cannot drift between components. The earlier attempt to keep the
originally specified `backend/app` and `ml/` directories and expose them as `neural_debugger.*`
(first via setuptools `package-dir`, then via two committed symlinks) was dropped: it shadowed the
editable install under uvicorn, needed reinstalls for new files, and symlinks are not portable
(Windows, some Docker/CI contexts). The conventional layout works identically in local development,
CI, wheels and Docker. Verified: a built wheel installed into a clean venv runs the app from an
unrelated directory; the Dockerfile's two-step `uv sync` was replayed outside Docker.
*Cost:* the directory tree differs from the one in the original project brief
(`backend/`, `ml/` at the repo root).

**Optional extras: `data` (ingestion + BM25 evaluation: httpx, beautifulsoup4, lxml, numpy) and `ml`
(torch etc., Milestone 3+).** The backend Docker image installs neither; CI installs `data`.
*Cost:* the production image will need the extras that retrieval requires once it ships.

**Single error envelope; validation errors omit submitted input.**
Users will paste private stack traces. FastAPI's default 422 body echoes the offending input, so it
is replaced with `loc`/`msg`/`type` only. Unhandled exceptions return a generic 500 and are logged
by exception *type* only (tracebacks can contain user data).

**Access logs record method, path, status, duration and request ID, nothing else.**
Query strings and bodies are excluded to honour "do not log raw stack traces by default".

**`/health` checks the database and returns 503 when it is down.**
A load balancer or orchestrator should stop routing to an instance that cannot serve requests.
The body still carries the diagnostic detail. The frontend treats a 503 with a valid body as
"degraded", distinct from "unreachable".

**Tests use real components, not mocks.**
Backend tests run the real app through Starlette's TestClient against a real SQLAlchemy engine; the
"database down" case uses a genuinely unreachable SQLite path. Frontend tests run against a real
local HTTP server so the real `fetch` and parsing code runs.

**Frontend: Next.js App Router + Vitest/Testing Library.** Following the Next.js docs bundled with
the installed version (v16), since APIs differ from earlier majors.

**CI has no secrets and no network calls to paid services.** Lint, types, tests and image builds only.

## Starlette `TestClient` warning (investigated, not suppressed broadly)

`starlette.testclient` (Starlette 1.10) imports `httpx2` if present and otherwise falls back to
`httpx` while emitting one `StarletteDeprecationWarning` at *import* time (`stacklevel=2`, so it is
attributed to `fastapi.testclient`). The warning only affects tests, and the fallback works. We keep
the well-known `httpx` dependency rather than adding a second HTTP client merely to silence a
warning (PyPI metadata for `httpx2` was checked: published from `github.com/pydantic/httpx2`,
author Tom Christie; it is the intended successor, so migrating later is a one-line change).

How it is handled in `pyproject.toml`: pytest runs with `filterwarnings = ["error", ...]` - every
warning fails the run - and a single `ignore` entry that matches this exact message, category
(`starlette.exceptions.StarletteDeprecationWarning`) and module (`fastapi.testclient`). Any other
warning, or this one from anywhere else, still fails the tests.

## M2 - Documentation retrieval and evaluation

**No database in Milestone 2.** The corpus is a deterministic JSONL file, BM25 is built in memory
(0.7 s for 4,995 chunks) and results are scored offline. A PostgreSQL schema adds nothing to the
measurements, and pgvector is explicitly deferred to dense retrieval (Milestone 4). Deviation from
the brief's original M2 list: *"Create PostgreSQL schema"* is moved to Milestones 4-6.

**Our own BM25 instead of a library.** ~80 lines, deterministic tie-breaking, filterable by
framework, section-collapsed results, and unit-tested against an independent hand-written
formula. It also makes the ranking function something we can explain line by line.

**Relevance is labeled at *section* level, and sections are the citable anchors.** Chunking
parameters can be tuned later without invalidating labels, and every label maps to a real
deep-link verified against the published page. API definitions (`dt id`) are sections of their own,
so `KeyError` or `Module.to` can be cited precisely.

**FastAPI is read from a pinned git commit, not crawled.** Clear MIT licence, exact version
(commit SHA), code includes resolvable, and no ambiguity about the site's crawler policy (its
`robots.txt` has no directives). Citation URLs still point at the live site; they were verified.

**Labels are honest about who made them.** The dataset has explicit provenance fields. Queries are
LLM-authored and unreviewed by a human, the official-FAQ slice is the only independently authored
one, and its lexical triviality is reported separately rather than averaged into the headline.
Alternatives considered: mining Stack Overflow accepted answers that link into the corpus
(independent human labels, but it needs a new external service and licence handling; offered as a
follow-up that needs approval), or synthetic labels (rejected as "ground truth").

**Configuration is declared before measuring, and tuned on dev only.** BM25 has no training, so
train + val serve as dev data; test was scored once the study was done. The paired-bootstrap study
found no significant improvement over the declared default, and the code-aware tokenizer's benefit
over plain tokens is *not* demonstrated (reported as such). A random-ranking floor is included so
metric values have a reference.

**Strict dataset invariants as tests, not conventions.** Corpus hash = manifest hash; unique chunk
ids (this caught a real parser bug); balanced code fences; URL shapes; labels reference existing
sections; materialised queries equal the YAML sources; splits equal the deterministic rule; no
near-duplicate crosses a split; provenance never overstates verification.

**Ruff `E501` is disabled in the linter** (the formatter wraps code at 100 columns, but cannot wrap
long string literals, docstring examples and error-message queries).
