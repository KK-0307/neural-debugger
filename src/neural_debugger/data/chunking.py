"""Structure-aware chunking.

* A chunk never crosses a section boundary, so each chunk maps to exactly one citable anchor.
* Blocks (paragraph / code / list / table) are the atomic unit: a code block is only split when
  it alone exceeds ``max_words``, and then only at blank lines (falling back to line breaks).
* Chunks are filled up to ``target_words``; a short trailing chunk is merged into its
  predecessor when the result stays within ``max_words * 1.25``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Block, Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])")


@dataclass(frozen=True)
class ChunkingConfig:
    target_words: int = 180
    max_words: int = 300
    min_tail_words: int = 30

    def as_dict(self) -> dict[str, int]:
        return {
            "target_words": self.target_words,
            "max_words": self.max_words,
            "min_tail_words": self.min_tail_words,
        }


def n_words(text: str) -> int:
    return len(text.split())


def render_block(block: Block) -> str:
    if block.kind == "code":
        return f"```{block.lang or ''}\n{block.text}\n```"
    return block.text


def _group(parts: list[str], sep: str, max_words: int) -> list[str]:
    groups: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for part in parts:
        w = n_words(part)
        if cur and cur_w + w > max_words:
            groups.append(sep.join(cur))
            cur, cur_w = [], 0
        cur.append(part)
        cur_w += w
    if cur:
        groups.append(sep.join(cur))
    return groups


def _hard_split(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def split_oversized(block: Block, max_words: int) -> list[Block]:
    """Split a block bigger than ``max_words`` while keeping its kind."""
    if n_words(block.text) <= max_words:
        return [block]
    if block.kind == "code":
        paragraphs = re.split(r"\n\s*\n", block.text)
        pieces: list[str] = []
        for p in paragraphs:
            if n_words(p) > max_words:
                pieces.extend(_group(p.split("\n"), "\n", max_words))
            else:
                pieces.append(p)
        texts = _group(pieces, "\n\n", max_words)
    elif block.kind in {"list", "table"}:
        texts = _group(block.text.split("\n"), "\n", max_words)
    else:
        sentences = _SENTENCE_END.split(block.text)
        texts = []
        for g in _group(sentences, " ", max_words):
            texts.extend([g] if n_words(g) <= max_words else _hard_split(g, max_words))
    return [Block(block.kind, t, block.lang) for t in texts if t.strip()]


def chunk_section(section: Section, config: ChunkingConfig = ChunkingConfig()) -> list[str]:
    """Return chunk bodies (markdown-ish text) for one section."""
    units = [u for b in section.blocks for u in split_oversized(b, config.max_words)]
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_w = 0
    for unit in units:
        w = n_words(unit.text)
        if cur and (cur_w >= config.target_words or cur_w + w > config.max_words):
            chunks.append(cur)
            cur, cur_w = [], 0
        cur.append(render_block(unit))
        cur_w += w
    if cur:
        chunks.append(cur)

    bodies = ["\n\n".join(c) for c in chunks]
    if len(bodies) >= 2 and n_words(bodies[-1]) < config.min_tail_words:
        merged = bodies[-2] + "\n\n" + bodies[-1]
        if n_words(merged) <= config.max_words * 1.25:
            bodies[-2:] = [merged]
    return bodies
