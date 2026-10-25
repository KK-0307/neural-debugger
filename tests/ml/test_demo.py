import json
from pathlib import Path

import pytest

from neural_debugger.data.schema import to_json_line
from neural_debugger.ml.retrieval import demo
from tests.helpers import make_chunk


def test_demo_cli_prints_ranked_results_with_urls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chunks = [
        make_chunk("keyerror", "A KeyError is raised when a dictionary key is not found."),
        make_chunk("other", "Completely unrelated text about asyncio event loops."),
    ]
    (tmp_path / "chunks.jsonl").write_text("\n".join(to_json_line(c) for c in chunks) + "\n")
    (tmp_path / "documents.jsonl").write_text("")
    assert (
        demo.main(["--corpus-dir", str(tmp_path), "--json", "-k", "2", "KeyError dictionary"]) == 0
    )
    out = json.loads(capsys.readouterr().out)
    hit = out[0]["hits"][0]
    assert hit["url"] == "https://example.test/python/1.0/page#keyerror"
    assert hit["section_id"].endswith("#keyerror") and out[0]["latency_ms"] >= 0
