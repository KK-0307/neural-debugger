"""Polite, cached HTTP fetching.

Rules enforced here: identify ourselves with a descriptive User-Agent, honour robots.txt,
wait between requests to the same host, retry with backoff, and never re-download a page
that is already in the local cache (``data/raw``). Cached bytes + sidecar metadata make
re-processing offline and deterministic.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from .schema import sha256_bytes

logger = logging.getLogger(__name__)


class FetchError(Exception):
    pass


class RobotsDisallowedError(FetchError):
    pass


@dataclass(frozen=True)
class RawPage:
    url: str
    final_url: str
    status: int
    content: bytes
    retrieved_at: str
    sha256: str
    from_cache: bool


def cache_path(raw_dir: Path, url: str) -> Path:
    parsed = urlparse(url)
    rel = parsed.path.lstrip("/") or "index"
    if rel.endswith("/"):
        rel += "index"
    return raw_dir / parsed.netloc / rel


class PoliteFetcher:
    def __init__(
        self,
        raw_dir: Path,
        user_agent: str,
        delay_seconds: float = 1.0,
        client: httpx.Client | None = None,
        max_retries: int = 3,
    ) -> None:
        self.raw_dir = raw_dir
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent}, follow_redirects=True, timeout=30.0
        )
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request: dict[str, float] = {}
        self.network_requests = 0
        self.bytes_downloaded = 0

    def close(self) -> None:
        self._client.close()

    # -- robots ---------------------------------------------------------------------------
    def _robots_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser = RobotFileParser()
            resp = self._get(f"{origin}/robots.txt", check_robots=False, allow_404=True)
            if resp is None:  # no robots.txt => everything allowed
                parser.parse([])
            else:
                parser.parse(resp.content.decode("utf-8", "replace").splitlines())
            self._robots[origin] = parser
        return self._robots[origin]

    def allowed(self, url: str) -> bool:
        return self._robots_for(url).can_fetch(self.user_agent, url)

    # -- fetching -------------------------------------------------------------------------
    def _wait(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            remaining = self.delay_seconds - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)

    def _get(
        self, url: str, check_robots: bool = True, allow_404: bool = False
    ) -> httpx.Response | None:
        if check_robots and not self.allowed(url):
            raise RobotsDisallowedError(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._wait(host)
            self._last_request[host] = time.monotonic()
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                self.network_requests += 1
                self.bytes_downloaded += len(resp.content)
                if resp.status_code == 404 and allow_404:
                    return None
                if resp.status_code == 200:
                    return resp
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise FetchError(f"{url}: HTTP {resp.status_code}")
                last_error = FetchError(f"{url}: HTTP {resp.status_code}")
            time.sleep(min(2**attempt * self.delay_seconds, 30))
        raise FetchError(f"giving up on {url}: {last_error}")

    def fetch(self, url: str) -> RawPage:
        path = cache_path(self.raw_dir, url)
        meta_path = path.with_name(path.name + ".meta.json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            content = path.read_bytes()
            return RawPage(
                url,
                meta["final_url"],
                meta["status"],
                content,
                meta["retrieved_at"],
                meta["sha256"],
                True,
            )
        resp = self._get(url)
        assert resp is not None
        retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        sha = sha256_bytes(resp.content)
        meta_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "final_url": str(resp.url),
                    "status": resp.status_code,
                    "retrieved_at": retrieved_at,
                    "sha256": sha,
                    "bytes": len(resp.content),
                },
                indent=2,
            )
        )
        return RawPage(url, str(resp.url), resp.status_code, resp.content, retrieved_at, sha, False)
