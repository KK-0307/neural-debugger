# Third-party documentation in `data/corpus/`

`data/corpus/chunks.jsonl` contains text chunks derived from the documentation below, kept with
their source URL, version and retrieval time. Each chunk links back to its original page.

| Source | Version | Licence | Origin |
|---|---|---|---|
| Python documentation | 3.12 | Python Software Foundation License v2 (https://docs.python.org/3.12/license.html) | https://docs.python.org/3.12/ |
| PyTorch documentation | 2.14 | BSD-3-Clause, `pytorch/pytorch` LICENSE | https://docs.pytorch.org/docs/2.14/ |
| FastAPI documentation | 0.162.0 (commit 7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4) | MIT, Copyright (c) 2018 Sebastian Ramirez | https://github.com/fastapi/fastapi |

## Supplement corpus (Milestone 5.2): `data/corpus_supplement/`

Four more pages of the **same official documentation sets and licences** as above, fetched politely (1 request/second,
`robots.txt` consulted, 6 requests in total) with the same pipeline into a separate directory, so the frozen base corpus is
untouched. Provenance (URL, retrieval time, content hash) is in `data/manifests/corpus_supplement_manifest.json`; the page list
and the reason each page was added are in `configs/sources_supplement.yaml`.

| Page | Version | Licence |
|---|---|---|
| `library/sys.html`, `library/sqlite3.html`, `whatsnew/3.12.html` on docs.python.org | 3.12 | Python Software Foundation License v2 |
| `generated/torch.compile.html` on docs.pytorch.org | 2.14 | BSD-3-Clause |

**Not ingested, deliberately:** public GitHub issues and Stack Overflow posts. Under GitHub's Terms of Service authors keep
ownership of what they post and other users are licensed only to use it "through the Service"; no licence to redistribute issue
text exists, and a repository's code licence does not cover its issue threads. Stack Overflow was not used at all (no scraping), and its own content terms were not reviewed in this project. See `docs/rag-coverage.md`.

The evaluation queries in `data/eval/` are original work of this project, except the `official_faq`
slice whose query text is the FAQ question headings of the Python and PyTorch documentation.
If you redistribute this corpus beyond research use, re-check each licence.

# Pretrained models used for retrieval (downloaded at run time, not stored in this repository)

| Model | Licence (per Hugging Face model card) | Notes |
|---|---|---|
| `thenlper/gte-small` | MIT | selected dense encoder |
| `BAAI/bge-small-en-v1.5` | MIT | compared on dev |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | compared on dev |
| `cross-encoder/ms-marco-MiniLM-L-12-v2`, `-L-6-v2` | Apache-2.0 | trained on MS MARCO, whose dataset terms are non-commercial research; unresolved for commercial deployment |

`data/embeddings/gte-small/` contains vectors computed from the corpus above by `thenlper/gte-small`
(revision recorded in its `meta.json`).
