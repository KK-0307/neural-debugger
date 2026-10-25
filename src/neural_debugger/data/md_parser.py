"""Parser for FastAPI's Markdown dialect (mkdocs-material).

Handles: ``# Heading { #explicit-id }``, fenced code, ``{* ../../docs_src/x.py hl[1] *}`` code
includes (resolved from the same git commit), ``/// note | Title`` admonitions and
``//// tab | Title`` tabs, and inline HTML noise.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .normalize import normalize_code, normalize_prose, slugify
from .schema import Block, ParsedPage, Section

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_EXPLICIT_ID = re.compile(r"\s*\{\s*#([\w.\-]+)\s*\}\s*$")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+#.\-]*)")
_INCLUDE = re.compile(r"^\{\*\s+(\S+)(?:\s+.*)?\*\}\s*$")
_ADMONITION = re.compile(r"^/{3,}\s*(\w+)?\s*(?:\|\s*(.*))?$")
_LIST = re.compile(r"^(\s*)([*+-]|\d+\.)\s+")
_TAG = re.compile(r"<[^>]+>")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_EMPH = re.compile(r"(\*\*|__)(.+?)\1")
_LANG_BY_SUFFIX = {
    ".py": "python",
    ".html": "html",
    ".js": "javascript",
    ".css": "css",
    ".txt": None,
}

IncludeResolver = Callable[[str], str | None]


def _inline(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _EMPH.sub(r"\2", text)
    return normalize_prose(_TAG.sub("", text))


def parse_fastapi_markdown(md: str, resolve_include: IncludeResolver) -> ParsedPage:
    lines = md.splitlines()
    if lines and lines[0].strip() == "---":  # front matter
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            lines = lines[end + 1 :]

    root = Section(anchor="", heading="", heading_path=[])
    sections = [root]
    stack: list[tuple[int, Section]] = []  # (level, section)
    current = root
    title = ""
    seen: set[str] = set()
    para: list[str] = []

    def flush() -> None:
        nonlocal para
        if not para:
            return
        first = para[0]
        text = "\n".join(para)
        if _LIST.match(first):
            kind = "list"
            body = "\n".join(_inline(ln) if not _LIST.match(ln) else _list_line(ln) for ln in para)
        elif first.lstrip().startswith("|"):
            kind = "table"
            body = "\n".join(
                " | ".join(_inline(c) for c in ln.strip().strip("|").split("|"))
                for ln in para
                if not re.fullmatch(r"[\s|:\-]+", ln)
            )
        else:
            kind = "paragraph"
            body = _inline(text.replace("\n", " "))
        if body.strip():
            current.blocks.append(Block(kind, body))  # type: ignore[arg-type]
        para = []

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            flush()
            indent, marker, lang = fence.groups()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker[0] * len(marker)):
                body.append(lines[i][len(indent) :] if lines[i].startswith(indent) else lines[i])
                i += 1
            i += 1  # closing fence
            code = normalize_code("\n".join(body))
            if code:
                current.blocks.append(Block("code", code, lang or None))
            continue
        inc = _INCLUDE.match(line.strip())
        if inc:
            flush()
            path = inc.group(1)
            source = resolve_include(path)
            if source is not None:
                suffix = "." + path.rsplit(".", 1)[-1] if "." in path else ""
                current.blocks.append(
                    Block("code", normalize_code(source), _LANG_BY_SUFFIX.get(suffix))
                )
            i += 1
            continue
        adm = _ADMONITION.match(line.strip())
        if adm and line.strip().startswith("///"):
            flush()
            kind, label = adm.group(1), adm.group(2)
            if kind:  # opening marker (bare "///" closes a block)
                head = {"tab": "Tab"}.get(kind, kind.capitalize())
                text = f"{head}: {_inline(label)}" if label else f"{head}:"
                current.blocks.append(Block("paragraph", text))
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            raw = heading.group(2)
            explicit = _EXPLICIT_ID.search(raw)
            anchor = explicit.group(1) if explicit else ""
            text = _inline(_EXPLICIT_ID.sub("", raw))
            anchor = anchor or slugify(text)
            base, n = anchor, 2
            while anchor in seen:
                anchor = f"{base}-{n}"
                n += 1
            seen.add(anchor)
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_path = stack[-1][1].heading_path if stack else []
            current = Section(anchor=anchor, heading=text, heading_path=[*parent_path, text])
            sections.append(current)
            stack.append((level, current))
            if level == 1 and not title:
                title = text
            i += 1
            continue
        if not line.strip() or line.startswith("::: "):
            flush()
        else:
            if para and _LIST.match(line) and not _LIST.match(para[0]):
                flush()
            para.append(line)
        i += 1
    flush()
    return ParsedPage(title=title, sections=[s for s in sections if s.blocks])


def _list_line(line: str) -> str:
    m = _LIST.match(line)
    assert m is not None
    indent = " " * (len(m.group(1)) // 2 * 2)
    marker = m.group(2)
    marker = "-" if marker in "*+-" else marker
    return f"{indent}{marker} {_inline(line[m.end() :])}"
