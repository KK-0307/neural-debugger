"""Which queries improved or regressed between two systems (per-query NDCG@10 deltas).

uv run python scripts/error_analysis.py data/eval/results/m3/dev_final bm25 hybrid-gte-tuned [N]
"""

import json
import sys
from pathlib import Path

d, ref, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 10


def load(name: str) -> dict[str, dict]:  # type: ignore[type-arg]
    return {
        r["id"]: r for r in map(json.loads, (d / f"{name}.rankings.jsonl").read_text().splitlines())
    }


queries = {
    r["id"]: r for r in map(json.loads, Path("data/eval/queries.jsonl").read_text().splitlines())
}
a, b = load(ref), load(new)
rows = sorted(
    ((b[i]["ndcg@10"] - a[i]["ndcg@10"], i) for i in a if queries[i]["slice"] != "official_faq"),
    reverse=True,
)


def show(title: str, items: list[tuple[float, str]]) -> None:
    print(f"\n## {title}")
    for delta, i in items:
        q = queries[i]
        rel = {s.split("/", 2)[2] for s in q["relevant"]}
        ra = next((k + 1 for k, s in enumerate(a[i]["ranking"]) if s in set(q["relevant"])), None)
        rb = next((k + 1 for k, s in enumerate(b[i]["ranking"]) if s in set(q["relevant"])), None)
        print(
            f"{i} [{q['slice']}/{q['framework']}] dNDCG {delta:+.2f} | first relevant rank {ref}={ra} {new}={rb}"
        )
        print(f"   q: {q['query'][:110]}")
        print(
            f"   want: {sorted(rel)[0][:80]}   top1 {ref}: {a[i]['ranking'][0].split('/', 2)[2][:55]}   top1 {new}: {b[i]['ranking'][0].split('/', 2)[2][:55]}"
        )


better = sum(delta > 1e-9 for delta, _ in rows)
worse = sum(delta < -1e-9 for delta, _ in rows)
print(f"{new} vs {ref}: improved {better}, regressed {worse}, tied {len(rows) - better - worse}")
show(f"Most improved ({new})", rows[:n])
show(f"Most regressed ({new})", [r for r in rows[::-1][:n]])
