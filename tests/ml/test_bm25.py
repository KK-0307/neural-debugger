import math
from collections import Counter

import numpy as np
import pytest

from neural_debugger.data.schema import Chunk
from neural_debugger.ml.retrieval.bm25 import BM25Config, BM25Retriever
from neural_debugger.ml.retrieval.tokenizer import TokenizerConfig, tokenize
from tests.helpers import make_chunk

CHUNKS = [
    make_chunk(
        "keyerror", "A KeyError is raised when a dictionary key is not found.", heading="KeyError"
    ),
    make_chunk(
        "indexerror",
        "An IndexError is raised when a sequence subscript is out of range.",
        heading="IndexError",
    ),
    make_chunk(
        "cuda",
        "Tensors must be on the same cuda device before an operation.",
        framework="pytorch",
        heading="CUDA",
    ),
    make_chunk("long", "dictionary " + "filler " * 200, heading="Long page"),
]


def reference_scores(
    chunks: list[Chunk], query: str, k1: float, b: float, cfg: TokenizerConfig
) -> list[float]:
    """Straight-from-the-formula BM25, written independently of the production code."""
    docs = [Counter(tokenize(c.index_text(), cfg)) for c in chunks]
    lens = [sum(d.values()) for d in docs]
    avg = sum(lens) / len(lens)
    n = len(docs)
    out = []
    for d, ln in zip(docs, lens, strict=True):
        s = 0.0
        for term in set(tokenize(query, cfg)):
            df = sum(1 for x in docs if term in x)
            if df == 0 or term not in d:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * d[term] * (k1 + 1) / (d[term] + k1 * (1 - b + b * ln / avg))
        out.append(s)
    return out


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (2.0, 0.3), (0.9, 1.0)])
def test_scores_match_reference_formula(k1: float, b: float) -> None:
    cfg = BM25Config(k1=k1, b=b)
    index = BM25Retriever(CHUNKS, cfg)
    for q in ["dictionary key not found", "cuda device tensors", "IndexError range"]:
        expected = reference_scores(CHUNKS, q, k1, b, cfg.tokenizer)
        np.testing.assert_allclose(index.score_chunks(q), expected, rtol=1e-12)


def test_relevant_chunk_ranks_first_with_metadata() -> None:
    hits = BM25Retriever(CHUNKS).search("what raises KeyError when key not found", k=3)
    assert hits[0].section_id.endswith("#keyerror")
    h = hits[0]
    assert h.rank == 1 and h.score > 0 and h.url.endswith("#keyerror")
    assert (h.framework, h.version) == ("python", "1.0") and h.snippet.startswith("A KeyError")
    assert [x.rank for x in hits] == list(range(1, len(hits) + 1))


def test_framework_filter_restricts_results() -> None:
    index = BM25Retriever(CHUNKS)
    assert {
        h.framework for h in index.search("device tensors dictionary", k=10, framework="pytorch")
    } == {"pytorch"}
    assert index.search("KeyError", framework="pytorch") == []


def test_results_are_collapsed_to_one_hit_per_section() -> None:
    chunks = [
        make_chunk("s", "tensor tensor tensor device", ordinal=0),
        make_chunk("s", "tensor device other words", ordinal=1),
        make_chunk("t", "tensor", ordinal=0),
    ]
    hits = BM25Retriever(chunks).search("tensor device", k=10)
    assert [h.section_id.split("#")[1] for h in hits].count("s") == 1
    assert hits[0].chunk_id.endswith("#s:0")  # the best chunk of the section represents it


def test_unknown_terms_return_no_results_and_k_is_respected() -> None:
    index = BM25Retriever(CHUNKS)
    assert index.search("zzzzqqqq") == []
    assert len(index.search("is raised when", k=2)) == 2


def test_ties_break_by_corpus_order_and_search_is_deterministic() -> None:
    chunks = [make_chunk(f"s{i}", "identical text here for ties") for i in range(5)]
    index = BM25Retriever(chunks)
    first = [h.chunk_id for h in index.search("identical text", k=5)]
    assert first == [c.chunk_id for c in chunks]
    assert first == [h.chunk_id for h in BM25Retriever(chunks).search("identical text", k=5)]


def test_length_normalisation_prefers_focused_chunk() -> None:
    hits = BM25Retriever(CHUNKS).search("dictionary", k=3)
    assert hits[0].section_id.endswith("#keyerror") or hits[0].section_id.endswith("#long")
    scores = {h.section_id.split("#")[1]: h.score for h in hits}
    assert (
        scores["keyerror"] > scores["long"]
    )  # same single match, much longer document scores less


def test_identifier_aware_tokenizer_beats_plain_on_subword_query() -> None:
    chunks = [
        make_chunk("target", "Raised when ModuleNotFoundError cannot import the module."),
        make_chunk("other", "Unrelated text about files and directories."),
    ]
    ident = BM25Retriever(chunks, BM25Config()).search("module not found error", k=2)
    plain = BM25Retriever(chunks, BM25Config(tokenizer=TokenizerConfig(mode="plain"))).search(
        "module not found error", k=2
    )
    assert ident and ident[0].section_id.endswith("#target")
    assert not plain or plain[0].score < ident[0].score
