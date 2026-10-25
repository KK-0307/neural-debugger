from neural_debugger.ml.retrieval.tokenizer import TokenizerConfig, s_stem, tokenize

PLAIN = TokenizerConfig(mode="plain")


def test_camel_case_exception_names_emit_whole_and_parts() -> None:
    assert set(tokenize("ModuleNotFoundError")) == {
        "modulenotfounderror",
        "module",
        "not",
        "found",
        "error",
    }
    assert set(tokenize("HTTPException")) == {"httpexception", "http", "exception"}


def test_dotted_identifiers_and_device_strings() -> None:
    assert set(tokenize("torch.nn.Module")) == {"torch.nn.module", "torch", "nn", "module"}
    assert set(tokenize("cuda:0")) == {"cuda:0", "cuda"}  # lone digit part dropped, whole kept


def test_numbers_versions_and_dunder_names() -> None:
    assert tokenize("3.12") == ["3.12"]
    assert tokenize("__init__") == ["init"]
    assert "mat1" in tokenize("mat1 and mat2")


def test_error_message_query_reaches_identifier_parts() -> None:
    toks = set(
        tokenize("RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu")
    )
    assert {
        "runtimeerror",
        "runtime",
        "error",
        "expected",
        "tensors",
        "device",
        "cuda:0",
        "cpu",
    } <= toks
    assert "the" not in toks and "to" not in toks


def test_stopwords_keep_negations_and_can_be_disabled() -> None:
    assert "not" in tokenize("this is not allowed")
    assert "the" in tokenize("the cat", TokenizerConfig(remove_stopwords=False))


def test_plain_mode_is_the_ablation_baseline() -> None:
    assert tokenize("ModuleNotFoundError torch.nn", PLAIN) == ["modulenotfounderror", "torch", "nn"]


def test_parts_only_variant_drops_whole_compound() -> None:
    cfg = TokenizerConfig(keep_compounds=False)
    assert set(tokenize("torch.nn.Module", cfg)) == {"torch", "nn", "module"}


def test_s_stemmer_only_touches_plural_endings() -> None:
    assert [s_stem(w) for w in ["tensors", "queries", "status", "glass", "gpu", "boxes"]] == [
        "tensor",
        "query",
        "status",
        "glass",
        "gpu",
        "boxe",
    ]
    assert tokenize("tensors", TokenizerConfig(stem=True)) == ["tensor"]


def test_tokenization_is_deterministic() -> None:
    text = "KeyError in dict.get() for torch.cuda.is_available() -> cuda:0"
    assert tokenize(text) == tokenize(text)
