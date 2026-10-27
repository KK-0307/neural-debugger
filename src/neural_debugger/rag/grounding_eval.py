"""Deterministic adversarial dataset and evaluation for the semantic grounding stage (Milestone 5.1).

Each item is ``(passage, claim, label, category)`` where the passage is a real corpus section and the claim is
either a verbatim piece of it (should be accepted) or a **programmatic mutation** of it (should not be
accepted silently). Mutations: negation, a changed number, a changed API signature, a fabricated identifier, an
antonym swap, and an appended causal claim that appears nowhere in the passage.

What this measures and what it does not:

* it is a **synthetic, agent-authored** benchmark built by rule: the labels are correct by construction (a
  mutated sentence really does say something different from the passage), but the claims are *not* natural LLM
  errors and the supported claims are verbatim/clausal rather than free paraphrase. Real paraphrase behaviour is
  inspected separately on live model output (``live_acceptance``), by agent review only;
* no human labelled anything; thresholds are chosen on the **dev half** (id-hash split) and reported on the
  **test half**, which is never used for choosing anything;
* numbers are ``accepted`` (silently passed), ``flagged`` (kept but visibly ``unverified``/``partial``) and
  ``rejected`` (dropped). For defective claims the headline error is *silent acceptance*; for supported claims
  it is *rejection*, and *flag rate* is the usability cost.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .semantic import (
    CrossEncoderNLI,
    GroundedSupportJudge,
    NLIScorer,
    SemanticConfig,
    SemanticGuard,
    parse_signature,
    split_sentences,
)
from .support import LexicalSupportJudge, SupportJudge, SupportVerdict

SEED = 20261026
SUPPORTED_QUOTE = (
    "supported_verbatim",
    "supported_clause",
    "supported_two_sentences",
    "supported_signature",
)
# light deterministic paraphrases: not substrings of the passage, so the verbatim-quote shortcut cannot fire
SUPPORTED_PARAPHRASE = ("supported_reordered", "supported_prefixed")
SUPPORTED = SUPPORTED_QUOTE + SUPPORTED_PARAPHRASE
DEFECTIVE = (
    "negation",
    "wrong_number",
    "wrong_signature",
    "fabricated_identifier",
    "antonym_swap",
    "unsupported_causal",
)
CATEGORIES = SUPPORTED + DEFECTIVE

_AUX = re.compile(
    r"\b(is|are|was|were|can|will|does|do|should|must|may|has|have|would|could)\b(?! not\b)(?!n't)"
)
_NOT = re.compile(
    r"\b(is|are|was|were|does|do|should|must|may|has|have|would|could|can|will)\s+not\b"
)
_STANDALONE_NUMBER = re.compile(r"(?<![\w.`/=-])(\d{1,3})(?![\w.`/%-])(?!\.\d)")
_CALL = re.compile(r"\b([A-Za-z_][\w.]*)\(([^()\n]{5,120})\)")
_BACKTICK_ID = re.compile(r"`([A-Za-z_][\w.]{2,40})`")
ANTONYMS = {
    "before": "after",
    "after": "before",
    "synchronous": "asynchronous",
    "asynchronous": "synchronous",
    "required": "optional",
    "optional": "required",
    "allowed": "forbidden",
    "increase": "decrease",
    "increases": "decreases",
    "decrease": "increase",
    "decreases": "increases",
    "enabled": "disabled",
    "disabled": "enabled",
    "first": "last",
    "last": "first",
    "always": "never",
    "never": "always",
    "immutable": "mutable",
    "mutable": "immutable",
    "same": "different",
    "different": "same",
    "faster": "slower",
    "slower": "faster",
    "larger": "smaller",
    "smaller": "larger",
    "raises": "returns",
    "returns": "raises",
    "inside": "outside",
    "outside": "inside",
    "valid": "invalid",
    "invalid": "valid",
    "True": "False",
    "False": "True",
}
_CONJUNCTIONS = frozenset(
    {"and", "but", "or", "for", "so", "nor", "yet", "if", "when", "then", "also", "however"}
)
_REORDER = re.compile(r"^(If|When|After|Before|Once|While)\s+(.+?),\s+(?:then\s+)?(.+?)\.$")
_LOWERABLE = frozenset(
    {"the", "a", "an", "this", "these", "you", "if", "when", "it", "in", "for", "to", "your"}
)
_ANTONYM = re.compile(r"\b(" + "|".join(sorted(ANTONYMS, key=len, reverse=True)) + r")\b")


@dataclass(frozen=True)
class GroundItem:
    id: str
    category: str
    label: str  # "supported" | "defective"
    passage: str
    claim: str
    source_sentence: str
    split: str  # "dev" | "test"
    framework: str


def split_of(item_id: str) -> str:
    return "dev" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def _prose_sentences(passage: str) -> list[str]:
    out = []
    for s in split_sentences(passage):
        words = s.split()
        alpha = sum(c.isalpha() or c == " " for c in s) / max(1, len(s))
        if (
            8 <= len(words) <= 40
            and alpha > 0.85
            and s[0].isupper()
            and s.endswith(".")
            and ">>>" not in s
            and "..." not in s
        ):
            out.append(s)
    return out


def _negate(s: str) -> str | None:
    if re.search(r"\b(not|never|no|cannot|n't)\b", s):
        m = _NOT.search(s)
        if not m:
            return None
        return s[: m.start()] + m.group(1) + s[m.end() :]
    m = _AUX.search(s)
    if not m:
        return None
    word = m.group(1)
    repl = "cannot" if word == "can" else f"{word} not"
    return s[: m.start()] + repl + s[m.end() :]


def _wrong_number(s: str, rng: random.Random) -> str | None:
    ms = list(_STANDALONE_NUMBER.finditer(s))
    if not ms:
        return None
    m = rng.choice(ms)
    n = int(m.group(1))
    new = n + rng.choice([1, 2, 3]) if n < 10 else n * 2
    return s[: m.start()] + str(new) + s[m.end() :]


def _mutate_signature(name: str, args: str, rng: random.Random) -> str | None:
    params = parse_signature(args)
    if len(params) < 2 or not all(re.fullmatch(r"[A-Za-z_]\w*", p[0]) for p in params):
        return None
    names = [p[0] for p in params]
    defaults = [p[1] for p in params]
    ops = []
    if len(set(names)) == len(names):
        ops += ["swap", "rename"]
    if any(d is not None for d in defaults):
        ops.append("default")
    if not ops:
        return None
    op = rng.choice(ops)
    if op == "swap":
        i = rng.randrange(len(names) - 1)
        names[i], names[i + 1] = names[i + 1], names[i]
        defaults[i], defaults[i + 1] = (
            defaults[i + 1],
            defaults[i],
        )  # defaults stay attached to positions
        if names == [p[0] for p in params]:
            return None
    elif op == "rename":
        i = rng.randrange(len(names))
        names[i] += "_x"
    else:
        i = rng.choice([k for k, d in enumerate(defaults) if d is not None])
        d = defaults[i]
        defaults[i] = (
            "False" if d == "True" else "True" if d == "False" else "None" if d != "None" else "0"
        )
    return f"{name}({', '.join(n + ('=' + d if d is not None else '') for n, d in zip(names, defaults, strict=True))})"


def _fabricate(identifier: str, vocabulary: set[str], rng: random.Random) -> str | None:
    head, _, last = identifier.rpartition(".")
    if len(last) < 3:
        return None
    for suffix in ("_async", "_all", "_fast", "_v2"):
        fake = f"{last}{suffix}"
        full = f"{head}.{fake}" if head else fake
        if fake.lower() not in vocabulary:
            return full
    return None


def generate(
    corpus: Any, per_category: int = 90, seed: int = SEED, max_chars: int = 1500
) -> list[GroundItem]:
    rng = random.Random(seed)
    section_ids = sorted(corpus.section_ids)
    rng.shuffle(section_ids)
    vocabulary = {
        w.lower() for c in corpus.chunks for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", c.content)
    }
    # round-robin over frameworks so python (3/4 of the corpus) does not dominate
    by_fw: dict[str, list[str]] = defaultdict(list)
    for sid in section_ids:
        by_fw[sid.split("/", 1)[0]].append(sid)
    ordered = [
        sid
        for group in itertools.zip_longest(*[by_fw[k] for k in sorted(by_fw)])
        for sid in group
        if sid
    ]
    counts: dict[str, int] = defaultdict(int)
    items: list[GroundItem] = []
    other_pool = [
        s for sid in ordered[:400] for s in _prose_sentences(corpus.section_text(sid)[:max_chars])
    ]

    def add(cat: str, sid: str, passage: str, claim: str, src: str, label: str) -> None:
        counts[cat] += 1
        iid = f"{cat}-{counts[cat]:03d}"
        items.append(
            GroundItem(iid, cat, label, passage, claim, src, split_of(iid), sid.split("/", 1)[0])
        )

    for sid in ordered:
        text = corpus.section_text(sid)[:max_chars]
        title = " > ".join(corpus.section_heading_path(sid)) or sid
        passage = f"{title}\n{text}"
        sents = _prose_sentences(text)
        if not sents:
            continue
        rs = random.Random(f"{seed}:{sid}")
        s = rs.choice(sents)
        done = {c: counts[c] >= per_category for c in CATEGORIES}
        if not done["supported_verbatim"]:
            add("supported_verbatim", sid, passage, s, s, "supported")
        if not done["supported_clause"]:
            parts = re.split(r",\s+|;\s+", s)
            if len(parts) >= 2 and len(parts[0].split()) >= 6:
                add("supported_clause", sid, passage, parts[0].rstrip(",;") + ".", s, "supported")
        if not done["supported_two_sentences"]:
            i = sents.index(s)
            if i + 1 < len(sents):
                two = f"{s} {sents[i + 1]}"
                add("supported_two_sentences", sid, passage, two, two, "supported")
        if not done["supported_reordered"]:
            for cand in sents:
                m = _REORDER.match(cand)
                if m and len(m.group(2).split()) >= 3 and len(m.group(3).split()) >= 3:
                    body = m.group(3)
                    add(
                        "supported_reordered",
                        sid,
                        passage,
                        f"{body[0].upper() + body[1:]} {m.group(1).lower()} {m.group(2)}.",
                        cand,
                        "supported",
                    )
                    break
        if not done["supported_prefixed"]:
            first = s.split()[0]
            body = s[0].lower() + s[1:] if first.lower() in _LOWERABLE else s
            add(
                "supported_prefixed",
                sid,
                passage,
                f"According to the documentation, {body}",
                s,
                "supported",
            )
        if not done["negation"]:
            negated = _negate(s)
            if negated and negated != s:
                add("negation", sid, passage, negated, s, "defective")
        if not done["wrong_number"]:
            ns = [x for x in sents if _STANDALONE_NUMBER.search(x)]
            if ns:
                src = rs.choice(ns)
                changed = _wrong_number(src, rs)
                if changed:
                    add("wrong_number", sid, passage, changed, src, "defective")
        if not done["antonym_swap"]:
            ans_ = [x for x in sents if _ANTONYM.search(x)]
            if ans_:
                src = rs.choice(ans_)
                ms = list(_ANTONYM.finditer(src))
                mm = rs.choice(ms)
                swapped = src[: mm.start()] + ANTONYMS[mm.group(1)] + src[mm.end() :]
                add("antonym_swap", sid, passage, swapped, src, "defective")
        if not (done["supported_signature"] and done["wrong_signature"]):
            calls = [
                (m.group(1), m.group(2))
                for m in _CALL.finditer(text)
                if "," in m.group(2) and "=" in m.group(2) or m.group(2).count(",") >= 2
            ]
            calls = [c for c in calls if _mutate_signature(c[0], c[1], random.Random(0))]
            if calls:
                name, args = rs.choice(calls)
                good = f"The signature is `{name}({args})`."
                mut = _mutate_signature(name, args, rs)
                if mut and not done["supported_signature"]:
                    add("supported_signature", sid, passage, good, good, "supported")
                if mut and not done["wrong_signature"]:
                    add(
                        "wrong_signature",
                        sid,
                        passage,
                        f"The signature is `{mut}`.",
                        good,
                        "defective",
                    )
        if not done["fabricated_identifier"]:
            cands = [
                (x, i)
                for x in sents
                for i in _BACKTICK_ID.findall(x)
                if "." in i or "_" in i or i.islower()
            ]
            if cands:
                src, ident = rs.choice(cands)
                fake = _fabricate(ident, vocabulary, rs)
                if fake:
                    add(
                        "fabricated_identifier",
                        sid,
                        passage,
                        src.replace(f"`{ident}`", f"`{fake}`"),
                        src,
                        "defective",
                    )
        if not done["unsupported_causal"]:
            other = rs.choice(other_pool)
            pterms = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", passage)}
            oterms = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", other)]
            first = other.split()[0].lower()
            if (
                oterms
                and first not in _CONJUNCTIONS
                and sum(t in pterms for t in oterms) / len(oterms) < 0.3
            ):
                tail = other[0].lower() + other[1:].rstrip(".")
                add(
                    "unsupported_causal",
                    sid,
                    passage,
                    f"{s.rstrip('.')}. This happens because {tail}.",
                    s,
                    "defective",
                )
        if all(counts[c] >= per_category for c in CATEGORIES):
            break
    return items


# ------------------------------------------------------------------------------------------ evaluation


class CachingNLI:
    """Memoise NLI scores so threshold sweeps re-use the (slow) model outputs. Also records latency."""

    def __init__(self, inner: NLIScorer) -> None:
        self.inner = inner
        self.cache: dict[tuple[str, str], dict[str, float]] = {}
        self.model_seconds = 0.0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        missing = [p for p in pairs if p not in self.cache]
        if missing:
            t = time.perf_counter()
            for p, sc in zip(missing, self.inner.score(missing), strict=True):
                self.cache[p] = sc
            self.model_seconds += time.perf_counter() - t
        return [self.cache[p] for p in pairs]


def outcome(v: SupportVerdict, drop_disputed: bool = False) -> str:
    """accepted | flagged | rejected, from what the pipeline would do with this verdict."""
    if v.label in ("unsupported", "contradicted") or (drop_disputed and v.semantic == "disputed"):
        return "rejected"
    if v.label in ("partial", "unchecked") or v.semantic in (
        "unverified",
        "disputed",
    ):  # "unconfirmed" is neutral
        return "flagged"
    return "accepted"


def run_system(
    items: Sequence[GroundItem], judge: SupportJudge, drop_disputed: bool = False
) -> dict[str, str]:
    return {it.id: outcome(judge.judge(it.claim, [it.passage]), drop_disputed) for it in items}


def score(items: Sequence[GroundItem], outcomes: dict[str, str]) -> dict[str, Any]:
    by_cat: dict[str, dict[str, int]] = {c: defaultdict(int) for c in CATEGORIES}
    for it in items:
        by_cat[it.category][outcomes[it.id]] += 1
    cats = {}
    for c, d in by_cat.items():
        n = sum(d.values())
        if n:
            cats[c] = {
                "n": n,
                "accepted": round(d["accepted"] / n, 3),
                "flagged": round(d["flagged"] / n, 3),
                "rejected": round(d["rejected"] / n, 3),
            }

    def pooled(names: Sequence[str], key: str) -> float | None:
        tot = sum(cats[c]["n"] for c in names if c in cats)
        return (
            round(sum(cats[c][key] * cats[c]["n"] for c in names if c in cats) / tot, 3)
            if tot
            else None
        )

    return {
        "per_category": cats,
        "defective_silently_accepted": pooled(DEFECTIVE, "accepted"),
        "defective_not_accepted_clean": (
            None
            if pooled(DEFECTIVE, "accepted") is None
            else round(1 - pooled(DEFECTIVE, "accepted"), 3)  # type: ignore[operator]
        ),
        "defective_rejected": pooled(DEFECTIVE, "rejected"),
        "supported_accepted": pooled(SUPPORTED, "accepted"),
        "supported_flagged": pooled(SUPPORTED, "flagged"),
        "supported_rejected": pooled(SUPPORTED, "rejected"),
        "supported_quote_accepted": pooled(SUPPORTED_QUOTE, "accepted"),
        "supported_quote_rejected": pooled(SUPPORTED_QUOTE, "rejected"),
        "supported_paraphrase_accepted": pooled(SUPPORTED_PARAPHRASE, "accepted"),
        "supported_paraphrase_flagged": pooled(SUPPORTED_PARAPHRASE, "flagged"),
        "supported_paraphrase_rejected": pooled(SUPPORTED_PARAPHRASE, "rejected"),
    }


def build_systems(nli: NLIScorer | None, cfg: SemanticConfig) -> dict[str, SupportJudge]:
    base = LexicalSupportJudge()
    systems: dict[str, SupportJudge] = {
        "lexical": base,
        "lexical+rules": GroundedSupportJudge(base, SemanticGuard(None, cfg)),
    }
    if nli is not None:
        systems["lexical+nli_only"] = GroundedSupportJudge(
            base, SemanticGuard(nli, SemanticConfig(**{**asdict(cfg), "use_rules": False}))
        )
        systems["lexical+rules+nli"] = GroundedSupportJudge(base, SemanticGuard(nli, cfg))
    return systems


def calibrate(
    dev: Sequence[GroundItem], nli: NLIScorer, max_false_reject: float = 0.05
) -> tuple[SemanticConfig, list[dict[str, Any]]]:
    """Choose thresholds on the dev half only.

    Rule: among configurations whose rejection of *supported* dev claims is <= ``max_false_reject``, minimise
    silent acceptance of defective dev claims, then the flag rate on supported claims.
    """
    grid = []
    for ent, con, ov in itertools.product(
        (0.5, 0.7, 0.85, 0.95), (0.5, 0.7, 0.85, 0.95), (0.3, 0.45, 0.6)
    ):
        cfg = SemanticConfig(min_match_overlap=ov, entail_threshold=ent, contradict_threshold=con)
        judge = GroundedSupportJudge(LexicalSupportJudge(), SemanticGuard(nli, cfg))
        sc = score(dev, run_system(dev, judge))
        grid.append(
            {
                "min_match_overlap": ov,
                "entail_threshold": ent,
                "contradict_threshold": con,
                "defective_silently_accepted": sc["defective_silently_accepted"],
                "supported_rejected": sc["supported_rejected"],
                "supported_flagged": sc["supported_flagged"],
            }
        )
    ok = [g for g in grid if (g["supported_rejected"] or 0) <= max_false_reject]
    if ok:
        best = min(
            ok,
            key=lambda g: (
                g["defective_silently_accepted"],
                g["supported_flagged"],
                -g["entail_threshold"],
            ),
        )
    else:  # nothing meets the false-rejection budget: take the least false-rejecting, then least permissive
        best = min(
            grid,
            key=lambda g: (
                g["supported_rejected"],
                g["defective_silently_accepted"],
                g["supported_flagged"],
            ),
        )
    cfg = SemanticConfig(
        min_match_overlap=best["min_match_overlap"],
        entail_threshold=best["entail_threshold"],
        contradict_threshold=best["contradict_threshold"],
    )
    return cfg, grid


def latency_ms(judge: SupportJudge, items: Sequence[GroundItem], n: int = 60) -> dict[str, float]:
    times = []
    for it in items[:n]:
        t = time.perf_counter()
        judge.judge(it.claim, [it.passage])
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "p50": round(statistics.median(times), 1),
        "p95": round(times[int(0.95 * (len(times) - 1))], 1),
    }


def _max_rss_mb() -> float:
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(
        peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1
    )  # bytes on macOS, KiB on Linux


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--per-category", type=int, default=90)
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out", type=Path, default=Path("data/eval/results/m5_1/semantic_grounding.json")
    )
    ap.add_argument(
        "--dataset-out", type=Path, default=Path("data/eval/rag/grounding_adversarial.jsonl")
    )
    args = ap.parse_args(argv)

    from neural_debugger.data.corpus import load_corpus

    corpus = load_corpus(args.corpus_dir)
    items = generate(corpus, args.per_category)
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_out.write_text("".join(json.dumps(asdict(i)) + "\n" for i in items))
    dev = [i for i in items if i.split == "dev"]
    test = [i for i in items if i.split == "test"]

    rss_before = _max_rss_mb()
    t_load = time.perf_counter()
    raw_nli = None if args.no_nli else CrossEncoderNLI(args.device)
    load_s = time.perf_counter() - t_load
    rss_after = _max_rss_mb()
    nli = None if raw_nli is None else CachingNLI(raw_nli)
    default_cfg = SemanticConfig()
    cfg = default_cfg
    grid: list[dict[str, Any]] = []
    if nli is not None:
        cfg, grid = calibrate(dev, nli)
    systems = build_systems(nli, cfg)
    # latency must be measured with an *uncached* model, otherwise the memoised sweep makes it look instant
    uncached = build_systems(raw_nli, cfg)

    report: dict[str, Any] = {
        "label": "synthetic, agent-authored adversarial set built by rule from the indexed corpus; "
        "thresholds chosen on dev only; no human labels",
        "seed": SEED,
        "dataset_sha256": hashlib.sha256(args.dataset_out.read_bytes()).hexdigest(),
        "n_items": {c: sum(i.category == c for i in items) for c in CATEGORIES},
        "n_dev": len(dev),
        "n_test": len(test),
        "nli_model": f"{CrossEncoderNLI.HF_ID}@{CrossEncoderNLI.REVISION}" if nli else None,
        "chosen_config": asdict(cfg),
        "default_config": asdict(default_cfg),
        "systems": {},
    }
    for name, judge in systems.items():
        report["systems"][name] = {
            "dev": score(dev, run_system(dev, judge)),
            "test": score(test, run_system(test, judge)),
            "latency_ms_per_claim_warm": latency_ms(uncached[name], test),
        }
    report["environment"] = {
        "device": args.device,
        "nli_load_seconds": round(load_s, 2) if nli else None,
        "peak_rss_mb_before_nli_load": rss_before,
        "peak_rss_mb_after_nli_load": rss_after,
    }
    if nli is not None:
        report["calibration_grid"] = grid
        report["nli_model_seconds_total"] = round(nli.model_seconds, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for name, r in report["systems"].items():
        t = r["test"]
        print(
            f"{name:20s} test: defective silently accepted {t['defective_silently_accepted']}, "
            f"rejected {t['defective_rejected']} | quotes accepted {t['supported_quote_accepted']} "
            f"rej {t['supported_quote_rejected']} | paraphrases accepted {t['supported_paraphrase_accepted']}, "
            f"flagged {t['supported_paraphrase_flagged']}, rejected {t['supported_paraphrase_rejected']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
