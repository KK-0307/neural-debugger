"""Training pairs for contrastive learning, with provenance and a hard leakage filter.

Synthetic pairs are generated from the *documentation* only (never from evaluation queries) and are
labelled ``synthetic_*``: they teach the model the vocabulary and structure of the docs but are not
ground truth. Three generators (deterministic given a seed):

* ``ict``     inverse cloze: a sentence/code line of a chunk becomes the query and is *removed* from the
              positive passage (10% of the time it is kept), the standard unsupervised retrieval recipe;
* ``heading`` the section heading becomes the query; the passage keeps its parent headings but not that
              heading;
* ``error``   error/exception lines found in code blocks (``KeyError: 'x'``) become queries and are
              removed from the passage.

Real labeled pairs (``real``) come only from the *train* split of the evaluation queries.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from neural_debugger.data.schema import Chunk
from neural_debugger.data.splits import jaccard, normalize_query, token_set

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])")
_ERROR_LINE = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*\.)?[A-Za-z_]\w*(?:Error|Exception|Warning|Exit|Interrupt)\b.*$"
)
_FENCE = re.compile(r"```[\w+-]*\n(.*?)```", re.S)


@dataclass(frozen=True)
class TrainPair:
    query: str
    doc_text: str
    chunk_idx: int  # corpus index of the positive chunk (negatives and masking use its section)
    section_id: str
    source: str  # synthetic_ict | synthetic_heading | synthetic_error | real


def doc_text(chunk: Chunk, content: str | None = None, drop_last_heading: bool = False) -> str:
    path = chunk.heading_path[:-1] if drop_last_heading else chunk.heading_path
    return f"{chunk.title}\n{' > '.join(path)}\n{chunk.content if content is None else content}"


def _units(content: str) -> list[str]:
    """Candidate query units: prose sentences and non-trivial code lines."""
    units: list[str] = []
    prose = _FENCE.sub("\n", content)
    for para in re.split(r"\n\s*\n", prose):
        for s in _SENTENCE.split(" ".join(para.split())):
            if 4 <= len(s.split()) <= 40:
                units.append(s.strip())
    for block in _FENCE.findall(content):
        for line in block.splitlines():
            line = line.strip().removeprefix(">>>").removeprefix("...").strip()
            if 3 <= len(line.split()) <= 30 and not line.startswith("#"):
                units.append(line)
    return units


def generate_synthetic(
    chunks: list[Chunk],
    seed: int = 0,
    ict_per_chunk: int = 3,
    keep_prob: float = 0.1,
    heading: bool = True,
    errors: bool = True,
    max_offset_chars: int = 1000,
) -> list[TrainPair]:
    rng = random.Random(seed)
    pairs: list[TrainPair] = []
    for idx, c in enumerate(chunks):
        # only units inside the visible prefix: the model truncates long chunks, so a sentence it
        # never sees cannot be learned from
        units = [u for u in _units(c.content) if 0 <= c.content.find(u) <= max_offset_chars]
        rng.shuffle(units)
        for u in units[:ict_per_chunk]:
            rest = c.content if rng.random() < keep_prob else c.content.replace(u, " ", 1)
            if len(rest.split()) >= 8:
                pairs.append(TrainPair(u, doc_text(c, rest), idx, c.section_id, "synthetic_ict"))
        if heading and c.ordinal == 0 and c.section_heading:
            q = c.section_heading.strip("` ")
            if len(q.split()) >= 1 and len(q) >= 3:
                pairs.append(
                    TrainPair(
                        q,
                        doc_text(c, drop_last_heading=True),
                        idx,
                        c.section_id,
                        "synthetic_heading",
                    )
                )
        if errors:
            for block in _FENCE.findall(c.content):
                for line in block.splitlines():
                    if _ERROR_LINE.match(line) and 3 <= len(line.split()) <= 40:
                        rest = c.content.replace(line, " ", 1)
                        pairs.append(
                            TrainPair(
                                line.strip(),
                                doc_text(c, rest),
                                idx,
                                c.section_id,
                                "synthetic_error",
                            )
                        )
    return pairs


@dataclass
class LeakageReport:
    dropped: int
    dropped_by_source: dict[str, int]
    examples: list[tuple[str, str]]


def filter_leakage(
    pairs: list[TrainPair], protected_queries: list[str], threshold: float = 0.6
) -> tuple[list[TrainPair], LeakageReport]:
    """Drop pairs whose query is identical or near-identical (token-set Jaccard >= threshold) to any
    protected evaluation query. Exact matches are checked via a hash set, near matches via an index
    on rare tokens, so the filter is O(pairs x candidates) rather than O(pairs x queries)."""
    exact = {normalize_query(q) for q in protected_queries}
    protected = [(q, token_set(q)) for q in protected_queries]
    by_token: dict[str, list[int]] = {}
    for i, (_, ts) in enumerate(protected):
        for t in ts:
            by_token.setdefault(t, []).append(i)
    kept: list[TrainPair] = []
    dropped = Counter[str]()
    examples: list[tuple[str, str]] = []
    for p in pairs:
        ts = token_set(p.query)
        hit = None
        if normalize_query(p.query) in exact:
            hit = p.query
        else:
            cand = {i for t in ts for i in by_token.get(t, ())}
            for i in cand:
                if jaccard(ts, protected[i][1]) >= threshold:
                    hit = protected[i][0]
                    break
        if hit is None:
            kept.append(p)
        else:
            dropped[p.source] += 1
            if len(examples) < 8:
                examples.append((p.query[:80], hit[:80]))
    return kept, LeakageReport(sum(dropped.values()), dict(dropped), examples)


def split_by_section(
    pairs: list[TrainPair], holdout_fraction: float, seed: int
) -> tuple[list[TrainPair], list[TrainPair]]:
    """Hold out whole sections (all their pairs) for synthetic validation: measures generalisation to
    sections the model never trained on, with no real query involved."""
    sections = sorted({p.section_id for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(sections)
    held = set(sections[: max(1, int(len(sections) * holdout_fraction))])
    return [p for p in pairs if p.section_id not in held], [
        p for p in pairs if p.section_id in held
    ]


def pairs_fingerprint(pairs: list[TrainPair]) -> str:
    h = hashlib.sha256()
    for p in pairs:
        h.update(json.dumps(asdict(p), sort_keys=True).encode())
    return h.hexdigest()


def summarize(pairs: list[TrainPair]) -> dict[str, Any]:
    c = Counter(p.source for p in pairs)
    return {
        "n_pairs": len(pairs),
        "by_source": dict(c),
        "n_sections": len({p.section_id for p in pairs}),
        "sha256": pairs_fingerprint(pairs),
    }
