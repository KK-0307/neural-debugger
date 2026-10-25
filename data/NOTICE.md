# Third-party documentation in `data/corpus/`

`data/corpus/chunks.jsonl` contains text chunks derived from the documentation below, kept with
their source URL, version and retrieval time. Each chunk links back to its original page.

| Source | Version | Licence | Origin |
|---|---|---|---|
| Python documentation | 3.12 | Python Software Foundation License v2 (https://docs.python.org/3.12/license.html) | https://docs.python.org/3.12/ |
| PyTorch documentation | 2.14 | BSD-3-Clause, `pytorch/pytorch` LICENSE | https://docs.pytorch.org/docs/2.14/ |
| FastAPI documentation | 0.162.0 (commit 7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4) | MIT, Copyright (c) 2018 Sebastian Ramirez | https://github.com/fastapi/fastapi |

The evaluation queries in `data/eval/` are original work of this project, except the `official_faq`
slice whose query text is the FAQ question headings of the Python and PyTorch documentation.
If you redistribute this corpus beyond research use, re-check each licence.
