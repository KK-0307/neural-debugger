"""Generate (or verify) persisted chunk embeddings for a configured model.

python -m neural_debugger.ml.retrieval.embed --model bge-small            # build if missing/stale
python -m neural_debugger.ml.retrieval.embed --model bge-small --device cpu --rebuild
python -m neural_debugger.ml.retrieval.embed --model bge-small --check    # exit 1 if stale/missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_debugger.data.corpus import load_corpus

from .embedding_store import StaleEmbeddingsError, load_embeddings
from .systems import SystemFactory, load_neural_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--model", required=True, help="key under embedding_models in configs/neural.yaml"
    )
    ap.add_argument("--config", type=Path, default=Path("configs/neural.yaml"))
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu) | cpu | mps | cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--check", action="store_true", help="only verify stored embeddings match")
    args = ap.parse_args(argv)

    corpus = load_corpus(args.corpus_dir)
    cfg = load_neural_config(args.config)
    factory = SystemFactory(corpus, cfg, device=args.device)
    encoder = factory.encoder(args.model)
    directory = factory.embedding_dir / args.model

    if args.check:
        try:
            emb = load_embeddings(directory, encoder, corpus.sha256, len(corpus.chunks))
        except (FileNotFoundError, StaleEmbeddingsError) as exc:
            print(f"STALE/MISSING: {exc}")
            return 1
        print(f"ok: {directory} ({emb.vectors.shape})")
        return 0

    from .embedding_store import ensure_embeddings

    emb = ensure_embeddings(
        directory, encoder, corpus.chunks, corpus.sha256, args.batch_size, args.rebuild
    )
    print(
        json.dumps(
            {k: emb.meta[k] for k in ("n_chunks", "embedding_seconds", "device", "bytes")}, indent=1
        )
    )
    print(f"stored in {directory} shape={emb.vectors.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
