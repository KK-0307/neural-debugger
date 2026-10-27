import statistics

import yaml

from neural_debugger.rag.coverage_eval import build_pipeline, diagnose_item
from neural_debugger.rag.evaluation import coverage_scan, load_items
from neural_debugger.rag.evidence import EvidenceConfig
from neural_debugger.rag.pipeline import PipelineConfig

pipe = build_pipeline(["data/corpus"], None, "hybrid-gte-tuned", "cpu")
items = load_items()
dev = [i for i in yaml.safe_load(open("data/eval/rag/coverage_dev.yaml"))["questions"]]
base_gold = [i for i in dev if i["gold"] and not any(g.get("supplement") for g in i["gold"])]
CFG = {
    "C0 k12 n5 d2": (12, EvidenceConfig()),
    "C1 k12 n6 d3": (12, EvidenceConfig(max_items=6, max_per_doc=3)),
    "C2 k16 n6 d3": (16, EvidenceConfig(max_items=6, max_per_doc=3)),
    "C3 k16 n7 d3 8k": (16, EvidenceConfig(max_items=7, max_per_doc=3, max_chars_total=8000)),
    "C4 k20 n8 d3 9k": (20, EvidenceConfig(max_items=8, max_per_doc=3, max_chars_total=9000)),
}
for name, (k, ev) in CFG.items():
    pipe.config = PipelineConfig(retrieve_k=k, evidence=ev)
    stages = [diagnose_item(pipe, it)["stage"] for it in base_gold]
    reach = sum(s == "reachable" for s in stages)
    shown = sum(s in ("reachable", "gate_blocked", "text_truncated") for s in stages)
    scan = coverage_scan(pipe, items)
    thr = 0.375
    tr = [c for it, c in scan if it.group == "authored" and it.split == "train"]
    fa = sum(c < thr for c in tr) / len(tr)
    un = [c for it, c in scan if it.group == "unanswerable"]
    ab = sum(c < thr for c in un) / len(un)
    chars = []
    for it in base_gold[:8]:
        p = pipe.prepare(it["q"], None)
        chars.append(sum(len(e.text) for e in p.evidence))
    oos = [diagnose_item(pipe, it)["gate"] == "none" for it in dev if it["kind"] == "abstain"]
    print(
        f"{name:16s} gold shown {shown}/{len(base_gold)} reachable {reach} | false-abstain(train authored)={fa:.3f} | unanswerable abstained={ab:.3f} (n={len(un)}) | dev-OOS gated {sum(oos)}/{len(oos)} | evidence chars mean {statistics.mean(chars):.0f}"
    )
