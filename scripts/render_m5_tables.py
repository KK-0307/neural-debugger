"""Markdown tables for Milestone 5 from data/eval/results/m5/*.json (never hand-copied).

uv run python scripts/render_m5_tables.py
"""

import json
from pathlib import Path

D = Path("data/eval/results/m5")
SYSTEMS = ["hybrid-gte-tuned", "hybrid-custom-tiny", "bm25"]


def load(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((D / f"rag_extractive_{name}.json").read_text())


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def retrieval_table() -> str:
    out = [
        "| retriever | split | n | gold in retrieved top-12 | gold in selected evidence | answer cites a gold section (answered) | false abstentions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in SYSTEMS:
        a = load(s)["automated"]["answerable_authored"]
        for split in ("all", "val"):
            r = a[split]
            out.append(
                f"| {s} | {split} | {r['n']} | {pct(r['retrieval']['gold_in_retrieved_top12'])} | "
                f"{pct(r['retrieval']['gold_in_evidence'])} | {pct(r['citations']['gold_cited_rate_of_answered'])} | {pct(r['abstained_rate'])} |"
            )
    return "\n".join(out)


def abstention_table() -> str:
    out = [
        "| retriever | out-of-scope library (n=22) abstained | off-topic (n=10) abstained | held-out half (n=16) abstained | false abstentions, dev answerable (n=161) | abstention F1 (all) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in SYSTEMS:
        a = load(s)["automated"]
        u = a["unanswerable"]
        n_ans = a["answerable_authored"]["all"]["n"] + a["answerable_faq"]["n"]
        fp = a["abstention"]["all"]["fp"]
        out.append(
            f"| {s} | {pct(u['out_of_scope_library']['abstained_rate'])} | {pct(u['off_topic']['abstained_rate'])} | "
            f"{pct(u['heldout']['abstained_rate'])} | {pct(fp / n_ans)} | {a['abstention']['all']['f1']:.3f} |"
        )
    return "\n".join(out)


def claims_table() -> str:
    r = load("hybrid-gte-tuned")["automated"]
    out = [
        "| slice | n | citation validity before filtering | claims | unsupported (lexical) | partial | dropped (no valid citation) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in (
        ("authored dev", r["answerable_authored"]["all"]),
        ("FAQ dev", r["answerable_faq"]),
    ):
        c = v["claims"]
        out.append(
            f"| {name} | {v['n']} | {pct(v['citations']['pre_validation_citation_validity'])} | {c['total_claims_before_filtering']} | "
            f"{pct(c['unsupported_rate'])} | {pct(c['partial_rate'])} | {pct(c['dropped_no_valid_citation_rate'])} |"
        )
    return "\n".join(out)


def latency_table() -> str:
    out = [
        "| retriever | retrieval p50 / p95 (ms) | generation p50 (ms) | validation p50 (ms) | total p50 / p95 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in SYSTEMS:
        t = load(s)["automated"]["latency_ms"]["all"]
        out.append(
            f"| {s} | {t['retrieval_ms']['p50']} / {t['retrieval_ms']['p95']} | {t['generation_ms']['p50']} | "
            f"{t['validation_ms']['p50']} | {t['total_ms']['p50']} / {t['total_ms']['p95']} |"
        )
    return "\n".join(out)


def fault_table() -> str:
    f = load("hybrid-gte-tuned")["fault_injection"]
    out = [
        f"Eligible answers: {f['eligible_answers']}.",
        "",
        "| injected fault | what it is | survived validation |",
        "|---|---|---:|",
    ]
    for k, v in f["faults"].items():
        out.append(
            f"| `{k}` | {v['description']} | {v['survived']} / {f['eligible_answers']} ({pct(v['survival_rate'])}) |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    for title, fn in (
        ("RETRIEVAL", retrieval_table),
        ("ABSTENTION", abstention_table),
        ("CLAIMS", claims_table),
        ("LATENCY", latency_table),
        ("FAULTS", fault_table),
    ):
        print(f"## {title}\n{fn()}\n")
