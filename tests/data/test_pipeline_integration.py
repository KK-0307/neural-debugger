"""End-to-end pipeline tests against a real local HTTP server and a real local git repository."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from neural_debugger.data.chunking import ChunkingConfig
from neural_debugger.data.corpus import load_corpus
from neural_debugger.data.fetch import FetchError, PoliteFetcher, RobotsDisallowedError
from neural_debugger.data.pipeline import build_corpus

UA = "NeuralDebuggerTest/0.0"
PAGE_A = """<html><body><div class="body" role="main"><section id="errors"><h1>Errors</h1>
<p>Handling a KeyError means catching the exception when a mapping key is missing from a dictionary.</p>
<section id="catching"><h2>Catching</h2><p>Use try and except to catch a KeyError and recover gracefully
in your program without crashing the whole process.</p></section></section></div></body></html>"""
PAGE_B = PAGE_A.replace("Errors", "More errors").replace(
    'id="errors"', 'id="more"'
)  # near-dup text
FLAKY = {"calls": 0}


class Handler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str]] = []

    def log_message(self, *args: object) -> None:  # silence
        pass

    def do_GET(self) -> None:
        Handler.requests.append((self.path, self.headers.get("User-Agent", "")))
        routes = {
            "/robots.txt": (200, b"User-agent: *\nDisallow: /private/\n"),
            "/docs/a.html": (200, PAGE_A.encode()),
            "/docs/b.html": (200, PAGE_B.encode()),
            "/private/secret.html": (200, b"secret"),
        }
        if self.path == "/docs/flaky.html":
            FLAKY["calls"] += 1
            status, body = (503, b"later") if FLAKY["calls"] == 1 else (200, PAGE_A.encode())
        else:
            status, body = routes.get(self.path, (404, b"nope"))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server() -> Iterator[str]:
    Handler.requests = []
    FLAKY["calls"] = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[str, str]:
    repo = tmp_path / "upstream"
    (repo / "docs/en/docs/tutorial").mkdir(parents=True)
    (repo / "docs_src").mkdir()
    (repo / "docs/en/docs/tutorial/errors.md").write_text(
        "# Errors { #errors }\n\nRaise an HTTPException to return an error response to the client "
        "with a status code and detail message.\n\n{* ../../docs_src/err.py *}\n"
    )
    (repo / "docs_src/err.py").write_text("raise HTTPException(status_code=404)\n")
    (repo / "LICENSE").write_text("MIT")
    git("init", "-q", "-b", "main", cwd=repo)
    git("config", "uploadpack.allowFilter", "true", cwd=repo)
    git("add", ".", cwd=repo)
    git("commit", "-q", "-m", "docs", cwd=repo)
    git("tag", "1.0.0", cwd=repo)
    return f"file://{repo}", git("rev-parse", "HEAD", cwd=repo)


def write_config(path: Path, base: str, repo: str, commit: str) -> None:
    path.write_text(
        f"""
corpus_id: test-corpus
user_agent: "{UA}"
request_delay_seconds: 0
sources:
  - name: pydocs
    framework: python
    version: "3.12"
    kind: sphinx_html
    base_url: {base}/docs/
    license: PSF
    pages: [a.html, b.html]
  - name: fa
    framework: fastapi
    version: "1.0.0"
    kind: git_markdown
    repo: {repo}
    ref: "1.0.0"
    commit: "{commit}"
    docs_dir: docs/en/docs
    site_url: https://fastapi.example/
    license: MIT
    include: ["tutorial/**/*.md"]
"""
    )


@pytest.fixture
def fetcher(tmp_path: Path) -> Iterator[PoliteFetcher]:
    f = PoliteFetcher(tmp_path / "raw", UA, delay_seconds=0)
    yield f
    f.close()


def test_fetcher_caches_identifies_itself_and_retries(server: str, fetcher: PoliteFetcher) -> None:
    f = fetcher
    first = f.fetch(f"{server}/docs/a.html")
    again = f.fetch(f"{server}/docs/a.html")
    assert not first.from_cache and again.from_cache and first.sha256 == again.sha256
    assert [p for p, _ in Handler.requests].count("/docs/a.html") == 1
    assert {ua for p, ua in Handler.requests if p == "/docs/a.html"} == {UA}
    assert f.fetch(f"{server}/docs/flaky.html").status == 200  # 503 then 200
    assert FLAKY["calls"] == 2
    with pytest.raises(FetchError):
        f.fetch(f"{server}/docs/missing.html")


def test_robots_disallow_is_honoured_without_requesting_the_page(
    server: str, fetcher: PoliteFetcher
) -> None:
    f = fetcher
    with pytest.raises(RobotsDisallowedError):
        f.fetch(f"{server}/private/secret.html")
    assert "/private/secret.html" not in [p for p, _ in Handler.requests]


def test_build_corpus_end_to_end_is_deterministic_and_carries_provenance(
    server: str, git_repo: tuple[str, str], tmp_path: Path
) -> None:
    repo, commit = git_repo
    cfg = tmp_path / "sources.yaml"
    write_config(cfg, server, repo, commit)
    raw = tmp_path / "raw"
    cc = ChunkingConfig(target_words=40, max_words=80, min_tail_words=5)

    m1 = build_corpus(cfg, raw, tmp_path / "out1", tmp_path / "m1.json", cc)
    n_requests = len(Handler.requests)
    m2 = build_corpus(cfg, raw, tmp_path / "out2", tmp_path / "m2.json", cc)  # fully cached
    assert len(Handler.requests) == n_requests  # no network on re-processing
    assert m1["corpus_chunks_sha256"] == m2["corpus_chunks_sha256"]
    assert (tmp_path / "out1/chunks.jsonl").read_bytes() == (
        tmp_path / "out2/chunks.jsonl"
    ).read_bytes()

    corpus = load_corpus(tmp_path / "out1")
    by_id = {c.chunk_id: c for c in corpus.chunks}
    c = by_id["python/3.12/a#catching:0"]
    assert (c.framework, c.version) == ("python", "3.12")
    assert c.url == f"{server}/docs/a.html#catching" and c.source_url == f"{server}/docs/a.html"
    assert c.heading_path == ["Errors", "Catching"] and c.ingested_at and len(c.content_hash) == 64
    fa = next(x for x in corpus.chunks if x.framework == "fastapi")
    assert fa.url == "https://fastapi.example/tutorial/errors/#errors"
    assert "raise HTTPException(status_code=404)" in fa.content  # include resolved from the commit

    # page b repeats page a's text: every one of its chunks is dropped, and each drop is recorded
    assert not any(x.doc_id.endswith("/b") for x in corpus.chunks)
    dropped = m1["dedup"]["n_exact_duplicates_dropped"] + m1["dedup"]["n_near_duplicates_dropped"]
    assert dropped == 2
    assert {d["duplicate_of"].split("#")[0] for d in m1["dedup_dropped"]} == {"python/3.12/a"}
    manifest = json.loads((tmp_path / "m1.json").read_text())
    assert manifest["sources"][1]["commit"] == commit
    assert {d["doc_id"] for d in manifest["documents"]} >= {
        "python/3.12/a",
        "fastapi/1.0.0/tutorial/errors",
    }
    assert all(len(d["raw_sha256"]) == 64 for d in manifest["documents"])


def test_wrong_pinned_commit_is_rejected(
    server: str, git_repo: tuple[str, str], tmp_path: Path
) -> None:
    repo, _ = git_repo
    cfg = tmp_path / "sources.yaml"
    write_config(cfg, server, repo, "0" * 40)
    with pytest.raises(Exception, match="pins"):
        build_corpus(cfg, tmp_path / "raw", tmp_path / "o", tmp_path / "m.json")


def test_failed_page_fails_the_build_loudly(
    server: str, git_repo: tuple[str, str], tmp_path: Path
) -> None:
    repo, commit = git_repo
    cfg = tmp_path / "sources.yaml"
    write_config(cfg, server, repo, commit)
    cfg.write_text(cfg.read_text().replace("[a.html, b.html]", "[a.html, gone.html]"))
    with pytest.raises(FetchError, match="gone.html"):
        build_corpus(cfg, tmp_path / "raw", tmp_path / "o", tmp_path / "m.json")
