"""Old (Milestone 5.2) gate vs the Milestone 5.3 combined gate on every *development* set, through the real pipeline code."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import yaml

from neural_debugger.rag.coverage_eval import build_pipeline
from neural_debugger.rag.evaluation import load_items
from neural_debugger.rag.sufficiency import SufficiencyConfig

pipe = build_pipeline(
    ["data/corpus", "data/corpus_supplement"],
    "data/embeddings_rag",
    "hybrid-gte-tuned",
    "cpu",
    "coverage",
)
new_cfg = pipe.config.sufficiency
old_cfg = (
    SufficiencyConfig()
)  # lexical only, threshold 0.375: the gate shipped through Milestone 5.2
items = load_items()
vd = yaml.safe_load(open("data/eval/rag/verbose_dev.yaml"))["questions"]
cd = yaml.safe_load(open("data/eval/rag/coverage_dev.yaml"))["questions"]
tb = yaml.safe_load(open("data/eval/rag/traceback_dev.yaml"))["questions"]
groups = {
    "answerable: train authored (calibration)": [
        i.query for i in items if i.group == "authored" and i.split == "train"
    ],
    "answerable: val authored": [
        i.query for i in items if i.group == "authored" and i.split == "val"
    ],
    "answerable: dev own+independent (verbose-ish)": [q["q"] for q in cd if q["kind"] == "answer"],
    "answerable: tracebacks": [q["q"] for q in tb],
    "answerable: verbose seeded (diagnosis set)": [q["q"] for q in vd if q["kind"] == "answer"],
    "out-of-scope: unanswerable calib half": [
        i.query for i in items if i.group == "unanswerable" and i.split == "calib"
    ],
    "out-of-scope: unanswerable held-out half": [
        i.query for i in items if i.group == "unanswerable" and i.split == "heldout"
    ],
    "out-of-scope: dev own+independent": [q["q"] for q in cd if q["kind"] == "abstain"],
    "out-of-scope: verbose seeded": [q["q"] for q in vd if q["kind"] == "abstain"],
}
out = {}
for name, texts in groups.items():
    row = {}
    for label, cfg in (("old", old_cfg), ("new", new_cfg)):
        pipe.config = dataclasses.replace(pipe.config, sufficiency=cfg)
        row[label] = sum(pipe.prepare(t, None).sufficiency.level == "none" for t in texts)
    out[name] = {"n": len(texts), "gate_abstains_old": row["old"], "gate_abstains_new": row["new"]}
    print(f"{name:48s} n={len(texts):3d}  gate abstains: old {row['old']:3d}  new {row['new']:3d}")
Path("data/eval/results/m5_3/gate_regression.json").write_text(
    json.dumps(
        {
            "label": "development sets only; old = lexical gate (0.375), new = combined gate (dense_weight 0.5, threshold 0.70)",
            "groups": out,
        },
        indent=1,
    )
    + "\n"
)
