"""Inspect the corpus by section (used when authoring and reviewing relevance labels).

python -m neural_debugger.data.browse find KEYWORD [KEYWORD ...] [--framework F] [--limit N]
python -m neural_debugger.data.browse show SECTION_ID [--chars N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .corpus import load_corpus


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["find", "show", "outline"])
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--framework")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--chars", type=int, default=1500)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    args = ap.parse_args(argv)
    corpus = load_corpus(args.corpus_dir)

    if args.command == "show":
        sid = args.terms[0]
        if sid not in corpus.section_ids:
            print(f"unknown section: {sid}")
            return 1
        print(f"{sid}\n{corpus.section_url(sid)}\n{' > '.join(corpus.section_heading_path(sid))}\n")
        print(corpus.section_text(sid)[: args.chars])
        return 0

    if args.command == "outline":
        for sid in sorted(corpus.section_ids):
            if sid.startswith(args.terms[0]):
                words = sum(c.n_words for c in corpus.section_chunks(sid))
                print(
                    f"{sid.split('#', 1)[-1]:55s} {words:5d}w  {corpus.section_heading_path(sid)[-1][:70]}"
                )
        return 0

    terms = [t.lower() for t in args.terms]
    shown = 0
    for sid in sorted(corpus.section_ids):
        chunks = corpus.section_chunks(sid)
        if args.framework and chunks[0].framework != args.framework:
            continue
        haystack = (" ".join(chunks[0].heading_path) + " " + corpus.section_text(sid)).lower()
        if all(t in haystack for t in terms):
            body = " ".join(corpus.section_text(sid).split())[:110]
            print(f"{sid}\n      {body}")
            shown += 1
            if shown >= args.limit:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
