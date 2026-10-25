# Data pipeline: ingestion, normalisation, chunking

Everything in this document is implemented in `src/neural_debugger/data/` and covered by tests in
`tests/data/`. Commands assume the repo root and `uv sync --extra data`.

## Corpus `nd-corpus-v1` (fixed and pinned)

| Source | Version pin | How it is obtained | Pages | Sections | Chunks |
|---|---|---|---:|---:|---:|
| Python docs | 3.12 (`docs.python.org/3.12/`) | HTTP, 1 request/s | 49 | 2,707 | 3,275 |
| PyTorch docs | 2.14 (`docs.pytorch.org/docs/2.14/`; `/stable/` redirects there) | HTTP, 1 request/s | 46 | 458 | 638 |
| FastAPI docs | 0.162.0, commit `7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4` | sparse `git clone` of the tagged commit | 112 | 989 | 1,082 |
| **Total** | | | **207** | **4,154** | **4,995** (521k words) |

The page list lives in [`configs/sources.yaml`](../configs/sources.yaml). It is deliberately small
(a "first dataset", not a crawl): the pages cover the errors and questions in the evaluation set
plus a broad body of surrounding material so retrieval is not trivial. Adding a page or changing a
version changes the corpus and must bump `corpus_id`.

### Access policy and licences

* A descriptive `User-Agent` is sent; `robots.txt` is fetched and honoured for every host
  (`fetch.py`, tested against a real local server: a disallowed path is never requested);
  one request per second per host; retry with back-off on 429/5xx; every response is cached under
  `data/raw/` (git-ignored) with a sidecar file holding URL, retrieval time, status and SHA-256, so
  re-processing never touches the network (verified: a rebuild makes 0 requests).
* Recon results at fetch time: `docs.python.org/3.12/` allowed (older versions are disallowed);
  `docs.pytorch.org` disallows only old versions and `master` (we use 2.14); the FastAPI site's
  `robots.txt` contains only a content-signal comment and no directives, so FastAPI text is taken
  from its MIT-licensed GitHub repository at a pinned commit instead of crawling the site. The site
  is contacted only to verify that citation links resolve (112 page requests).
* Download size for the initial build: 97 requests, about 24 MB of decompressed HTML/markdown
  (Python 8.7 MB, PyTorch 15.1 MB, FastAPI 0.7 MB); less on the wire (gzip).
* Licences (recorded per document in `documents.jsonl`): Python docs - PSF License v2; PyTorch -
  BSD-3-Clause (the docs are built from `pytorch/pytorch`); FastAPI - MIT. The processed corpus
  (`data/corpus/`) is committed so results are reproducible without re-crawling; see
  [`data/NOTICE.md`](../data/NOTICE.md) for attribution. Pages are redistributed only as derived,
  attributed text chunks that link back to the original page.

## Pipeline

```
sources.yaml ─▶ fetch (cached, polite) ─▶ parse (Sphinx HTML | FastAPI markdown)
             ─▶ normalise ─▶ chunk ─▶ de-duplicate ─▶ data/corpus/{documents,chunks}.jsonl
                                                      data/manifests/corpus_manifest.json
```

Run: `make corpus` (= `python -m neural_debugger.data.ingest build`); verify citation URLs with
`make verify-urls`.

### Parsing (structure-aware)

* **Sphinx pages** (Python, PyTorch; `html_parser.py`): the main content container is extracted
  (navigation, permalinks, `[source]` links dropped); `<section id>` elements become sections with
  heading paths; API definitions (`dl.py/c/...` -> `dt id=...`) become *individually citable
  sections* (`KeyError`, `torch.nn.Module.to`); code blocks keep their language; lists and tables are
  rendered to text; KaTeX duplicates are collapsed to the TeX source.
* **FastAPI markdown** (`md_parser.py`): explicit heading ids (`## Title { #id }`) are used as
  anchors; fenced code is verbatim (so `#` comments are not mistaken for headings);
  `{* ../../docs_src/x.py *}` code includes are resolved from the same git commit; admonitions and
  tabs become labelled paragraphs.
* Definitions without an `id` cannot be deep-linked, so they stay inside their parent section. (A
  bug where such entries became anchorless, colliding sections was found by the corpus-integrity
  test and fixed; the pipeline now fails loudly on duplicate section ids.)

### Normalisation

Prose: Unicode NFC, typographic quotes/ellipsis to ASCII, non-breaking/zero-width/soft-hyphen
characters removed, whitespace collapsed. Code: only line endings, tabs and trailing whitespace are
touched; identifiers, indentation and literals are preserved. Inline code keeps its backticks.

### Chunking

Chunks never cross a section boundary (so every chunk maps to exactly one citable anchor). Blocks
are atomic: a code block is split only if it alone exceeds the maximum, then at blank lines.
Parameters: target 180 words, maximum 300, short tails (< 30 words) merged into the previous chunk.
Median chunk is 86 words (mean 104, p95 233, max 424). Code is rendered as fenced blocks inside
`content`; 3,455 chunks contain code and all fences are balanced (tested).

### De-duplication

Exact duplicates (case/whitespace-insensitive) and near-duplicates (5-word-shingle Jaccard >= 0.85,
found with MinHash-LSH then verified exactly; chunks under 25 words are exempt) are removed, keeping
the first occurrence. Result: 29 exact + 13 near duplicates dropped from 5,037 chunks. Every
decision is recorded (`dedup_dropped` in the corpus manifest: dropped chunk, kept chunk, kind,
Jaccard).

### Identifiers and provenance fields

| Field | Example / meaning |
|---|---|
| `doc_id` | `python/3.12/library/exceptions` |
| `section_id` | `python/3.12/library/exceptions#KeyError` (relevance labels use this) |
| `chunk_id` | `python/3.12/library/exceptions#KeyError:0` |
| `title`, `section_heading`, `heading_path` | page title, section heading, ancestors |
| `framework`, `version` | `python`, `3.12` |
| `source_url`, `url` | page URL; citation deep link (`...#KeyError`) |
| `content`, `content_hash` | chunk text, SHA-256 of it |
| `ingested_at` | UTC time the source was retrieved (from fetch metadata, so rebuilds are byte-identical) |
| documents: `raw_sha256`, `license` | hash of the exact fetched bytes; licence string |

### Determinism and URL validity (verified)

* Two builds from the same raw cache are byte-identical (`test_build_corpus_end_to_end_...`), and
  the corpus SHA-256 is recorded in the manifest and checked by a test.
* **All 207 cited pages resolve and all 4,154 citation anchors exist** in the published pages
  (`data/manifests/url_verification.json`, produced by `make verify-urls`): Python 2,707/2,707,
  PyTorch 458/458, FastAPI 989/989 (anchors computed from the git commit were checked against the
  live site, which is not pinned to the same commit).

## Limitations

* Published pages change over time; the raw cache and the committed corpus are the reproducibility
  anchor, not the live URLs. A citation may therefore drift from a later site build.
* PyTorch is represented by 46 pages (14 notes, 7 API overview pages, 25 per-symbol pages), so most
  of its API is not indexed; many realistic queries have no answer in this small corpus (the
  evaluation set only contains queries whose answers are indexed).
* No GitHub issues / Stack Overflow content yet.
* Windows checkouts are not tested (the earlier symlink layout was removed).
