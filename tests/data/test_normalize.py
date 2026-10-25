from neural_debugger.data.normalize import dedup_key, normalize_code, normalize_prose, slugify


def test_prose_normalisation_fixes_typography_but_keeps_identifiers() -> None:
    raw = "It’s a “KeyError” for  `torch.nn.Module`¶ ​\r\n\r\n\r\n\r\nNext"
    assert normalize_prose(raw) == 'It\'s a "KeyError" for `torch.nn.Module`\n\nNext'


def test_code_normalisation_preserves_indentation_and_content() -> None:
    raw = "\n\n  def f(x):\n\treturn x  \n\n    # “keep”  two  spaces\n\n"
    out = normalize_code(raw)
    assert out == "  def f(x):\n    return x\n\n    # “keep”  two  spaces"


def test_dedup_key_ignores_case_and_whitespace_only() -> None:
    assert dedup_key("Hello   World\n") == dedup_key("hello world")
    assert dedup_key("hello world") != dedup_key("hello worlds")


def test_slugify_matches_mkdocs_style() -> None:
    assert slugify("Use `HTTPException` now!") == "use-httpexception-now"
    assert slugify("Path  Parameters -- and more") == "path-parameters-and-more"
