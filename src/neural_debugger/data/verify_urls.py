"""Verify that every citation URL in the corpus resolves, down to the anchor.

For each document the *published page* (the URL we cite, not a local file) is fetched through
the polite, cached fetcher and every section anchor is looked up as an ``id=`` in the HTML.
Sphinx sources were parsed from exactly these pages, so they pass by construction; FastAPI is
parsed from a pinned git commit but cited at the live site, so its check is a real test.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import load_config
from .corpus import load_corpus
from .fetch import FetchError, PoliteFetcher


@dataclass
class FrameworkReport:
    pages: int = 0
    pages_ok: int = 0
    anchors: int = 0
    anchors_found: int = 0
    page_failures: list[str] = field(default_factory=list)
    missing_anchors: list[str] = field(default_factory=list)


_ID = re.compile(r"""\bid=["']([^"']+)["']""")


def run_verify_urls(config_path: Path, raw_dir: Path, corpus_dir: Path, manifest_path: Path) -> int:
    cfg = load_config(config_path)
    corpus = load_corpus(corpus_dir)
    anchors_by_doc: dict[str, set[str]] = defaultdict(set)
    for c in corpus.chunks:
        if "#" in c.url:
            anchors_by_doc[c.doc_id].add(c.url.split("#", 1)[1])

    fetcher = PoliteFetcher(raw_dir, cfg.user_agent, cfg.request_delay_seconds)
    report: dict[str, FrameworkReport] = defaultdict(FrameworkReport)
    try:
        for doc in corpus.documents:
            try:
                page = fetcher.fetch(doc.source_url)
            except FetchError as exc:
                entry = report[doc.framework]
                entry.pages += 1
                entry.page_failures.append(f"{doc.source_url}: {exc}")
                continue
            ids = set(_ID.findall(page.content.decode("utf-8", "replace")))
            wanted = anchors_by_doc[doc.doc_id]
            missing = sorted(wanted - ids)
            entry = report[doc.framework]
            entry.pages += 1
            entry.pages_ok += 1
            entry.anchors += len(wanted)
            entry.anchors_found += len(wanted) - len(missing)
            entry.missing_anchors.extend(f"{doc.source_url}#{a}" for a in missing)
    finally:
        fetcher.close()

    ok = True
    for framework, entry in sorted(report.items()):
        print(
            f"{framework:8s} pages {entry.pages_ok}/{entry.pages}  "
            f"anchors {entry.anchors_found}/{entry.anchors}  "
            f"(page failures: {len(entry.page_failures)}, "
            f"missing anchors: {len(entry.missing_anchors)})"
        )
        ok &= not entry.page_failures
    out = manifest_path.with_name("url_verification.json")
    out.write_text(
        json.dumps({k: asdict(v) for k, v in report.items()}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"details: {out}")
    return 0 if ok else 1
