from neural_debugger.data.md_parser import parse_fastapi_markdown

MD = """# Handling Errors { #handling-errors }

Intro paragraph with a [link](https://example.com) and **bold**.

## Use `HTTPException` { #use-httpexception }

```python
# this is a comment, not a heading
## neither is this
raise HTTPException(status_code=404)
```

{* ../../docs_src/handling_errors/tutorial001.py hl[1] *}

/// note | Technical Details

Admonition body.

///

* item one
* item two

| a | b |
|---|---|
| 1 | 2 |

## Heading without explicit id

Text.
"""


def resolve(path: str) -> str | None:
    return {"../../docs_src/handling_errors/tutorial001.py": "print('included')\n"}.get(path)


def test_headings_use_explicit_ids_and_slug_fallback() -> None:
    page = parse_fastapi_markdown(MD, resolve)
    assert page.title == "Handling Errors"
    anchors = [s.anchor for s in page.sections]
    assert anchors == ["handling-errors", "use-httpexception", "heading-without-explicit-id"]
    assert page.sections[1].heading_path == ["Handling Errors", "Use `HTTPException`"]


def test_fenced_code_is_verbatim_and_hash_lines_inside_are_not_headings() -> None:
    page = parse_fastapi_markdown(MD, resolve)
    code = [b for b in page.sections[1].blocks if b.kind == "code"]
    assert code[0].lang == "python"
    assert code[0].text.splitlines()[1] == "## neither is this"


def test_includes_resolved_and_admonitions_lists_tables() -> None:
    page = parse_fastapi_markdown(MD, resolve)
    blocks = page.sections[1].blocks
    assert any(b.kind == "code" and b.text == "print('included')" for b in blocks)
    assert any(b.text == "Note: Technical Details" for b in blocks)
    assert any(b.kind == "list" and b.text == "- item one\n- item two" for b in blocks)
    assert any(b.kind == "table" and b.text == "a | b\n1 | 2" for b in blocks)
    assert page.sections[0].blocks[0].text == "Intro paragraph with a link and bold."


def test_unresolvable_include_is_skipped_not_fabricated() -> None:
    page = parse_fastapi_markdown(
        "# T\n\n{* ../../docs_src/missing.py *}\n\ntext\n", lambda p: None
    )
    assert [b.text for b in page.sections[0].blocks] == ["text"]
