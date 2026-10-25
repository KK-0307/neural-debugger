"""Command-line entry point for corpus ingestion.

python -m neural_debugger.data.ingest build            # fetch (cached) + parse + write corpus
python -m neural_debugger.data.ingest verify-urls      # check cited URLs/anchors resolve
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .chunking import ChunkingConfig
from .config import load_config
from .fetch import PoliteFetcher
from .pipeline import build_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["build", "verify-urls"])
    parser.add_argument("--config", type=Path, default=Path("configs/sources.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/corpus_manifest.json")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "build":
        cfg = load_config(args.config)
        fetcher = PoliteFetcher(args.raw_dir, cfg.user_agent, cfg.request_delay_seconds)
        try:
            manifest = build_corpus(
                args.config, args.raw_dir, args.out_dir, args.manifest, ChunkingConfig(), fetcher
            )
        finally:
            fetcher.close()
        print(json.dumps(manifest["totals"], indent=1))
        print(json.dumps(manifest["dedup"], indent=1))
        print(
            f"network requests this run: {fetcher.network_requests}, "
            f"downloaded: {fetcher.bytes_downloaded / 1e6:.1f} MB (cache hits cost nothing)"
        )
        return 0

    from .verify_urls import run_verify_urls

    return int(run_verify_urls(args.config, args.raw_dir, args.out_dir, args.manifest))


if __name__ == "__main__":
    sys.exit(main())
