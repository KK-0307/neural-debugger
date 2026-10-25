from neural_debugger.data.chunking import ChunkingConfig, chunk_section, n_words
from neural_debugger.data.schema import Block, Section

CFG = ChunkingConfig(target_words=40, max_words=80, min_tail_words=10)


def section(blocks: list[Block]) -> Section:
    return Section(anchor="s", heading="S", heading_path=["S"], blocks=blocks)


def para(n: int, tag: str = "w") -> Block:
    return Block("paragraph", " ".join(f"{tag}{i}" for i in range(n)))


def test_small_section_is_one_chunk() -> None:
    chunks = chunk_section(section([para(10), para(10)]), CFG)
    assert len(chunks) == 1 and n_words(chunks[0]) == 20


def test_chunks_respect_max_words_and_cover_all_text() -> None:
    blocks = [para(30, f"a{i}x") for i in range(10)]
    chunks = chunk_section(section(blocks), CFG)
    assert len(chunks) > 1
    assert all(n_words(c) <= CFG.max_words * 1.25 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"a{i}x29" in joined for i in range(10))


def test_code_block_is_kept_whole_and_fenced() -> None:
    code = Block("code", "\n".join(f"line_{i} = {i}" for i in range(12)), "python")
    chunks = chunk_section(section([para(25), code, para(25)]), CFG)
    holder = [c for c in chunks if "line_0 = 0" in c]
    assert len(holder) == 1 and "line_11 = 11" in holder[0]
    assert "```python\nline_0 = 0" in holder[0]


def test_oversized_code_splits_at_blank_lines_without_losing_lines() -> None:
    groups = ["\n".join(f"g{g}_line{i}" for i in range(30)) for g in range(4)]
    code = Block("code", "\n\n".join(groups), "python")
    chunks = chunk_section(section([code]), CFG)
    assert len(chunks) >= 2
    for g in range(4):  # each blank-line-separated group stays contiguous in one chunk
        assert sum(f"g{g}_line0\n" in c and f"g{g}_line29" in c for c in chunks) == 1
    assert sum(c.count("_line") for c in chunks) == 120


def test_tiny_tail_is_merged_into_previous_chunk() -> None:
    chunks = chunk_section(section([para(45), para(5, "tail")]), CFG)
    assert len(chunks) == 1 and "tail4" in chunks[0]


def test_empty_section_yields_nothing_and_output_is_deterministic() -> None:
    assert chunk_section(section([]), CFG) == []
    blocks = [para(33, f"d{i}x") for i in range(6)]
    assert chunk_section(section(blocks), CFG) == chunk_section(section(blocks), CFG)
