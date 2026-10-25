import random

from neural_debugger.data.dedupe import dedupe_chunks, jaccard, near_duplicate_pairs, shingles
from tests.helpers import make_chunk


def words(seed: int, n: int = 80) -> str:
    rng = random.Random(seed)
    vocab = [f"token{i}" for i in range(500)]
    return " ".join(rng.choice(vocab) for _ in range(n))


def test_exact_duplicates_dropped_first_wins() -> None:
    a = make_chunk("a", words(1))
    b = make_chunk("b", "  " + words(1).upper() + "\n")  # differs only by case/whitespace
    c = make_chunk("c", words(2))
    kept, report = dedupe_chunks([a, b, c])
    assert [x.section_id for x in kept] == [a.section_id, c.section_id]
    assert report.n_exact_duplicates == 1
    assert report.dropped[0]["duplicate_of"] == a.chunk_id


def test_near_duplicates_dropped_but_distinct_text_kept() -> None:
    base = words(3)
    edited = base.replace("token", "tok", 1) + " extra"  # tiny edit, Jaccard well above 0.85
    kept, report = dedupe_chunks(
        [make_chunk("a", base), make_chunk("b", edited), make_chunk("c", words(4))]
    )
    assert [c.section_id.split("#")[1] for c in kept] == ["a", "c"]
    assert report.n_near_duplicates == 1


def test_short_chunks_are_not_near_deduplicated() -> None:
    kept, _ = dedupe_chunks(
        [make_chunk("a", "short text here"), make_chunk("b", "short text there")]
    )
    assert len(kept) == 2


def test_lsh_finds_exactly_the_pairs_brute_force_finds_at_high_threshold() -> None:
    texts = [words(i) for i in range(12)]
    texts += [texts[0] + " x", texts[5].replace("token", "tok", 2)]
    sets = [shingles(t) for t in texts]
    brute = {
        (i, j)
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
        if jaccard(sets[i], sets[j]) >= 0.85
    }
    found = {(i, j) for i, j, _ in near_duplicate_pairs(sets, 0.85)}
    assert brute and found == brute
