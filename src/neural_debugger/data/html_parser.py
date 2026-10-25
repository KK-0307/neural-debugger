"""Structure-aware parser for Sphinx-generated documentation (docs.python.org, docs.pytorch.org).

Output is a list of *sections*, each with the anchor id used by the published page, so every
retrieved passage can be cited with a working deep link. Two things count as a section:

* a ``<section id=...>`` with its heading (prose pages), and
* an API definition ``<dt id=...>`` of a ``dl.py`` / ``dl.c`` entry (functions, classes,
  methods, exceptions), so ``KeyError`` or ``torch.nn.Module.to`` are individually citable.

Code blocks are kept verbatim as ``code`` blocks.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PageElement, Tag

from .normalize import normalize_code, normalize_prose
from .schema import Block, ParsedPage, Section

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_DROP_SELECTORS = (
    "script, style, nav, button, .headerlink, .toctree-wrapper, .katex-html, .MathJax_Preview, "
    ".linenos, .copybtn, .sphinx-codeautolink-a, .sd-tab-label, .bd-header-article, "
    ".prev-next-area, .viewcode-link"
)
_INLINE_CODE = {"code", "tt", "kbd", "samp"}
_API_DL_CLASSES = {"py", "c", "cpp", "js", "std"}
_ADMONITION_TITLE = "admonition-title"


def _main_container(soup: BeautifulSoup) -> Tag:
    for selector in ("article.bd-article", "div.body[role=main]", "div[role=main]", "main"):
        found = soup.select_one(selector)
        if found is not None:
            return found
    raise ValueError("no main content container found")


def _classes(el: Tag) -> set[str]:
    cls = el.get("class")
    if cls is None:
        return set()
    return set(cls.split()) if isinstance(cls, str) else set(cls)


def _inline_text(el: PageElement, in_code: bool = False) -> str:
    """Flatten inline content to text; inline code keeps backticks; math keeps its TeX."""
    if isinstance(el, NavigableString):
        return str(el)
    assert isinstance(el, Tag)
    classes = _classes(el)
    if el.name == "br":
        return "\n"
    if "katex-mathml" in classes:  # keep the TeX source, drop the duplicated rendered MathML
        tex = el.find("annotation", attrs={"encoding": "application/x-tex"})
        return f" {tex.get_text()} " if tex is not None else ""
    inner = "".join(_inline_text(c, in_code or el.name in _INLINE_CODE) for c in el.children)
    if el.name in _INLINE_CODE and not in_code:
        stripped = inner.strip()
        return f"`{stripped}`" if stripped else ""
    return inner


def _text(el: Tag) -> str:
    return normalize_prose(_inline_text(el))


def _flat_text(el: Tag) -> str:
    return normalize_prose(re.sub(r"\s+", " ", _inline_text(el)))


def _code_lang(pre: Tag) -> str | None:
    for parent in list(pre.parents)[:3]:  # Sphinx wraps: div.highlight-LANG > div.highlight > pre
        for c in _classes(parent):
            if c.startswith("highlight-"):
                lang = c.removeprefix("highlight-")
                return None if lang in {"default", "none"} else lang
    return None


def _render_list(el: Tag, depth: int = 0) -> list[str]:
    lines: list[str] = []
    ordered = el.name == "ol"
    for i, li in enumerate(el.find_all("li", recursive=False), start=1):
        marker = f"{i}." if ordered else "-"
        own: list[str] = []
        nested: list[str] = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                nested.extend(_render_list(child, depth + 1))
            elif isinstance(child, Tag) and child.name == "pre":
                own.append("\n" + normalize_code(child.get_text()))
            else:
                own.append(_inline_text(child))
        text = normalize_prose(re.sub(r"[ \t\r\f\v]*\n[ \t\r\f\v]*(?!\n)", " ", "".join(own)))
        lines.append("  " * depth + f"{marker} {text}")
        lines.extend(nested)
    return lines


def _render_table(el: Tag) -> str:
    rows = []
    for tr in el.find_all("tr"):
        cells = [_flat_text(c) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


class _Walker:
    def __init__(self) -> None:
        self.sections: list[Section] = []
        self._stack: list[Section] = []
        self._seen_anchors: set[str] = set()
        self._root = Section(anchor="", heading="", heading_path=[])
        self.sections.append(self._root)
        self._stack.append(self._root)

    @property
    def current(self) -> Section:
        return self._stack[-1]

    def open(self, anchor: str, heading: str) -> Section:
        base = anchor
        n = 2
        while anchor and anchor in self._seen_anchors:  # ids are unique in Sphinx; be defensive
            anchor = f"{base}-{n}"
            n += 1
        self._seen_anchors.add(anchor)
        parent_path = self.current.heading_path
        section = Section(anchor=anchor, heading=heading, heading_path=[*parent_path, heading])
        self.sections.append(section)
        self._stack.append(section)
        return section

    def close(self) -> None:
        self._stack.pop()

    def add(self, block: Block) -> None:
        if block.text.strip():
            self.current.blocks.append(block)

    # -- element dispatch --------------------------------------------------------------------
    def visit(self, el: PageElement) -> None:
        if isinstance(el, NavigableString):
            text = normalize_prose(str(el))
            if text:
                self.add(Block("paragraph", text))
            return
        assert isinstance(el, Tag)
        name = el.name
        classes = _classes(el)
        if name == "section":
            self._visit_section(el)
        elif name in _HEADINGS:
            return  # headings are consumed by their enclosing <section>
        elif name == "dl":
            self._visit_dl(el, classes)
        elif name == "p":
            if _ADMONITION_TITLE in classes:
                self.add(Block("paragraph", _flat_text(el) + ":"))
            else:
                self.add(Block("paragraph", _text(el)))
        elif name == "pre":
            self.add(Block("code", normalize_code(el.get_text()), _code_lang(el)))
        elif name in {"ul", "ol"}:
            self.add(Block("list", "\n".join(_render_list(el))))
        elif name == "table":
            self.add(Block("table", _render_table(el)))
        elif name in {"img", "svg", "script", "style", "hr"}:
            return
        else:
            for child in el.children:
                self.visit(child)

    def _visit_section(self, el: Tag) -> None:
        heading_el = next(
            (c for c in el.children if isinstance(c, Tag) and c.name in _HEADINGS), None
        )
        heading = _flat_text(heading_el) if heading_el is not None else ""
        self.open(str(el.get("id", "")), heading)
        for child in el.children:
            if child is not heading_el:
                self.visit(child)
        self.close()

    def _visit_dl(self, el: Tag, classes: set[str]) -> None:
        if classes & _API_DL_CLASSES and "field-list" not in classes:
            self._visit_api_entries(el)
        else:
            self._visit_definition_list(el)

    def _visit_api_entries(self, el: Tag) -> None:
        children = [c for c in el.children if isinstance(c, Tag)]
        i = 0
        while i < len(children):
            child = children[i]
            if child.name == "dt":
                dts = [child]
                while i + 1 < len(children) and children[i + 1].name == "dt":
                    i += 1
                    dts.append(children[i])
                dd = (
                    children[i + 1]
                    if i + 1 < len(children) and children[i + 1].name == "dd"
                    else None
                )
                i += 1 if dd is not None else 0
                primary = next((d for d in dts if d.get("id")), dts[0])
                anchor = str(primary.get("id", ""))
                if anchor:
                    self.open(anchor, anchor)
                # Entries without an id cannot be deep-linked, so they stay inside the section
                # that contains them (a section needs a unique anchor to be citable).
                for d in dts:
                    sig = _flat_text(d)
                    if sig:
                        self.add(Block("code", sig, None))
                if dd is not None:
                    for c in dd.children:
                        self.visit(c)
                if anchor:
                    self.close()
            i += 1

    def _visit_definition_list(self, el: Tag) -> None:
        """Glossary-style and field lists (Parameters/Returns/...): keep term + description."""
        for child in el.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "dt":
                self.add(Block("paragraph", _flat_text(child).rstrip(":") + ":"))
            elif child.name == "dd":
                for c in child.children:
                    self.visit(c)


def parse_sphinx_html(html: str) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    root = _main_container(soup)
    for dead in root.select(_DROP_SELECTORS):
        dead.decompose()
    # "[source]" / "[docs]" links to rendered source files carry no documentation text.
    for a in root.find_all("a"):
        if a.get_text(strip=True) in {"[source]", "[docs]"}:
            a.decompose()
    h1 = root.find("h1")
    title = _flat_text(h1) if h1 is not None else ""
    if not title and soup.title is not None:
        title = soup.title.get_text().split("—")[0].split("&mdash;")[0].strip()
    walker = _Walker()
    for child in root.children:
        walker.visit(child)
    sections = [s for s in walker.sections if s.blocks]
    return ParsedPage(title=title, sections=sections)


def iter_blocks(page: ParsedPage) -> Iterator[Block]:
    for s in page.sections:
        yield from s.blocks
