from neural_debugger.data.html_parser import parse_sphinx_html

HTML = """
<html><head><title>Errors &mdash; Docs</title></head><body>
<nav>SITE NAVIGATION SHOULD NOT APPEAR</nav>
<div class="body" role="main">
 <section id="errors">
  <h1>8. Errors<a class="headerlink" href="#errors">¶</a></h1>
  <p>Intro with <code class="docutils literal">KeyError</code> and a <a href="x">link</a>.</p>
  <section id="syntax">
   <h2>8.1. Syntax</h2>
   <div class="highlight-python3 notranslate"><div class="highlight"><pre>
<span></span><span class="k">def</span> <span class="nf">f</span><span class="p">():</span>
    <span class="k">return</span> <span class="mi">1</span>
</pre></div></div>
   <ul><li>first<ul><li>nested</li></ul></li><li>second</li></ul>
   <table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
   <p>Math <span class="katex-mathml"><math><semantics><mrow>RENDERED</mrow>
     <annotation encoding="application/x-tex">x^2</annotation></semantics></math></span></p>
  </section>
  <dl class="py exception">
   <dt class="sig sig-object py" id="KeyError"><em class="property">exception </em><span class="sig-name">KeyError</span><a class="reference external" href="s">[source]</a><a class="headerlink" href="#KeyError">¶</a></dt>
   <dd><p>Raised when a mapping key is not found.</p>
    <dl class="py method">
     <dt class="sig sig-object py" id="KeyError.args"><span class="sig-name">args</span></dt>
     <dd><p>The key.</p></dd>
    </dl>
   </dd>
  </dl>
 </section>
</div></body></html>
"""


def test_sections_anchors_and_heading_paths() -> None:
    page = parse_sphinx_html(HTML)
    assert page.title == "8. Errors"
    by_anchor = {s.anchor: s for s in page.sections}
    assert set(by_anchor) == {"errors", "syntax", "KeyError", "KeyError.args"}
    assert by_anchor["syntax"].heading_path == ["8. Errors", "8.1. Syntax"]
    # API entries nest under the section that contains them, and methods under their class
    assert by_anchor["KeyError"].heading_path == ["8. Errors", "KeyError"]
    assert by_anchor["KeyError.args"].heading_path == ["8. Errors", "KeyError", "KeyError.args"]


def test_inline_code_code_blocks_lists_tables_math() -> None:
    page = parse_sphinx_html(HTML)
    by_anchor = {s.anchor: s for s in page.sections}
    intro = by_anchor["errors"].blocks[0]
    assert intro.text == "Intro with `KeyError` and a link."
    syntax = by_anchor["syntax"].blocks
    code = next(b for b in syntax if b.kind == "code")
    assert code.text == "def f():\n    return 1" and code.lang == "python3"
    assert next(b for b in syntax if b.kind == "list").text == "- first\n  - nested\n- second"
    assert next(b for b in syntax if b.kind == "table").text == "a | b\n1 | 2"
    math = next(b for b in syntax if b.text.startswith("Math"))
    assert "x^2" in math.text and "RENDERED" not in math.text


def test_navigation_permalinks_and_source_links_removed() -> None:
    page = parse_sphinx_html(HTML)
    text = " ".join(b.text for s in page.sections for b in s.blocks) + " ".join(
        s.heading for s in page.sections
    )
    assert "NAVIGATION" not in text and "[source]" not in text and "¶" not in text


def test_api_entry_keeps_signature_and_description() -> None:
    page = parse_sphinx_html(HTML)
    key_error = next(s for s in page.sections if s.anchor == "KeyError")
    assert key_error.blocks[0].kind == "code" and key_error.blocks[0].text == "exception KeyError"
    assert key_error.blocks[1].text == "Raised when a mapping key is not found."


def test_api_entries_without_an_id_stay_in_their_parent_section() -> None:
    """Regression: anchorless <dt> used to become anchorless sections whose ids collided."""
    html = """<div class="body" role="main"><section id="more-on-lists"><h1>More on Lists</h1>
    <dl class="py method"><dt class="sig sig-object py"><span class="sig-name">list.append</span>(x)</dt>
    <dd><p>Add an item to the end.</p></dd></dl>
    <dl class="py method"><dt class="sig sig-object py"><span class="sig-name">list.extend</span>(it)</dt>
    <dd><p>Extend the list.</p></dd></dl></section></div>"""
    page = parse_sphinx_html(html)
    assert [s.anchor for s in page.sections] == ["more-on-lists"]
    texts = [b.text for b in page.sections[0].blocks]
    assert "list.append(x)" in texts and "Add an item to the end." in texts
    assert "list.extend(it)" in texts and "Extend the list." in texts
