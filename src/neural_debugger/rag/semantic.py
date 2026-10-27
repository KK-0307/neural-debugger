"""Semantic grounding: does the cited text *mean* what the claim says? (Milestone 5.1)

The lexical judge (``support.py``) only measures vocabulary overlap, so a claim with its meaning inverted, a
changed number or a mutated API signature looks "supported". This stage adds, in order:

1. **Deterministic checks** against the best-matching sentence of the cited passages:
   * numbers: a claim number that differs from the matching sentence's numbers -> contradicted;
   * API signatures ``name(a, b=1)``: parameter names/order/defaults compared with the documented signature
     -> contradicted on a mismatch;
   * identifiers (backticked names, dotted paths) that appear nowhere in the cited text -> unverified;
   * negation polarity: a matched sentence with the opposite polarity -> contradicted.
2. **NLI** (optional, ``NLIScorer``): a cross-encoder trained on MNLI-style data scores (premise, claim) for the
   top candidate sentence windows. Its *contradiction* output is trusted only when the premise window is
   topically matched to the claim, because NLI models call unrelated sentences "contradictions".

Outcome per claim: ``supported``; ``contradicted`` (hard evidence: a changed number or API signature - the claim
is removed); ``disputed`` (soft signal: an NLI contradiction or a negation-polarity difference - the claim is
kept and flagged, because on real model output these signals were measured to fire on correct claims);
``unverified`` (a specific finding: an identifier or number of the claim is absent from the cited text);
or ``unconfirmed`` (nothing contradicts the claim, but it could not be positively confirmed - neutral, it does
not change the claim, because on real model paraphrases a small NLI model confirms almost nothing and flagging
every claim would carry no information). The reasons and the premise window used are kept so the
judgement is inspectable. These are model/heuristic
judgements, **not ground truth**; ``unverified`` is a first-class, visible result, not an acceptance.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .support import SupportJudge, SupportVerdict, identifier_present, identifiers
from .text import term_set

Verdict = Literal["supported", "contradicted", "disputed", "unverified", "unconfirmed"]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])|\n+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?![\w])")
_NEGATION = re.compile(
    r"\b(not|no|never|cannot|without|neither|nor|unable|none|nothing|disable[sd]?|n't)\b|n't\b",
    re.I,
)
_CALL = re.compile(r"([A-Za-z_][\w.]*)\(([^()]*)\)")
_MARKER = re.compile(r"\[\s*E\d+(?:\s*,\s*E\d+)*\s*\]", re.I)
_STOP_FOR_MATCH = frozenset({"not", "no", "never", "cannot"})
_ATTACHED_NUMBER = re.compile(
    r"[A-Za-z/:._-](\d{2,})"
)  # 3000 in localhost:3000, 2 in v2 is too short
_BACKTICK_SPAN = re.compile(r"`([^`\n]{2,120})`")


@dataclass
class SemanticResult:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    window: str | None = None  # the premise sentence(s) the decision rests on
    nli: dict[str, float] | None = (
        None  # {"entailment", "contradiction", "neutral"} for that window
    )
    match_overlap: float = 0.0

    @property
    def uncertain(self) -> bool:
        return self.verdict == "unverified"


class NLIScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """For each (premise, hypothesis): probabilities for entailment / contradiction / neutral."""
        ...


class CrossEncoderNLI:
    """``cross-encoder/nli-deberta-v3-xsmall`` (Apache-2.0, 283 MB, pinned revision) on CPU/MPS."""

    HF_ID = "cross-encoder/nli-deberta-v3-xsmall"
    REVISION = "a1508764"

    def __init__(self, device: str = "cpu", model: Any = None) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.HF_ID, revision=self.REVISION, device=device)
        self._model = model
        labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
        self._idx = {v: k for k, v in labels.items()}

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        probs = self._model.predict(list(pairs), apply_softmax=True, batch_size=16)
        return [
            {k: float(row[self._idx[k]]) for k in ("entailment", "contradiction", "neutral")}
            for row in probs
        ]


@dataclass(frozen=True)
class SemanticConfig:
    """Thresholds chosen on the *dev half* of the synthetic adversarial set (``grounding_eval.calibrate``:
    lowest silent acceptance of defective claims subject to <= 5% rejection of supported ones); the test half
    was not used. They are validated only on synthetic minimal-edit mutations and a 41-claim real sample."""

    min_match_overlap: float = (
        0.6  # claim-term coverage for a sentence to count as "the matching premise"
    )
    entail_threshold: float = 0.95
    contradict_threshold: float = 0.85
    max_windows: int = 4
    use_rules: bool = True


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("`", "").split())


def split_sentences(text: str) -> list[str]:
    out = []
    for s in _SENT.split(text):
        s = " ".join(s.split())
        if len(s.split()) >= 3:
            out.append(s)
    return out


def _content_terms(text: str) -> set[str]:
    return {t for t in term_set(text) if t not in _STOP_FOR_MATCH and not t.isdigit()}


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _neg_count(text: str) -> int:
    return len(_NEGATION.findall(text))


def _split_args(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_signature(args: str) -> list[tuple[str, str | None]]:
    out = []
    for a in _split_args(args):
        a = a.strip().lstrip("*")
        if not a or a in ("/", "..."):
            continue
        name, _, default = a.partition("=")
        name = name.split(":")[0].strip()
        out.append((name, default.strip() or None))
    return out


def signature_conflict(claim: str, passages_text: str) -> str | None:
    """Compare ``f(a, b=1)`` forms in the claim with the documented form of the same callable."""
    docs: dict[str, list[list[tuple[str, str | None]]]] = {}
    for m in _CALL.finditer(passages_text):
        docs.setdefault(m.group(1).split(".")[-1], []).append(parse_signature(m.group(2)))
    for m in _CALL.finditer(claim):
        short = m.group(1).split(".")[-1]
        mine = parse_signature(m.group(2))
        if short not in docs or not mine:
            continue
        candidates = [d for d in docs[short] if d]
        if not candidates:
            continue
        if any(
            [p[0] for p in d] == [p[0] for p in mine]
            and all(
                (a[1] is None or b[1] is None or a[1] == b[1]) for a, b in zip(d, mine, strict=True)
            )
            for d in candidates
        ):
            continue
        # A claim may cite only some of the parameters: it conflicts if a named parameter/default disagrees
        # or the order of the shared names differs, or it names a parameter the documented form lacks.
        for d in candidates:
            dn = [p[0] for p in d]
            mn = [p[0] for p in mine]
            extra = [n for n in mn if n not in dn]
            shared_doc = [n for n in dn if n in mn]
            shared_mine = [n for n in mn if n in dn]
            bad_default = [
                n
                for n, dv in mine
                if n in dn
                and dv is not None
                and next(x[1] for x in d if x[0] == n) not in (None, dv)
            ]
            if not extra and shared_doc == shared_mine and not bad_default:
                break
        else:
            return f"signature of {m.group(1)}(...) differs from the documentation"
    return None


class SemanticGuard:
    name = "semantic"

    def __init__(self, nli: NLIScorer | None = None, config: SemanticConfig | None = None) -> None:
        self.nli = nli
        self.cfg = config or SemanticConfig()

    # ------------------------------------------------------------------ public

    def check(self, claim: str, passages: Sequence[str], trusted: str = "") -> SemanticResult:
        """Judge ``claim`` against the cited ``passages``.

        A claim that is a verbatim quote of the cited text is supported. Otherwise each sentence of the claim is
        judged on its own and the results are combined: any contradiction contradicts the claim; it is supported
        only if every sentence is.
        """
        claim = " ".join(_MARKER.sub(" ", claim).split())
        text = "\n".join(passages)
        if _norm(claim).rstrip(".") and _norm(claim).rstrip(".") in _norm(text):
            return SemanticResult(
                "supported", ["verbatim quote of the cited text"], None, None, 1.0
            )
        sentences = [s for p in passages for s in split_sentences(p)]
        parts = split_sentences(claim) or [claim]
        results = [self._check_part(part, sentences, text, trusted.lower()) for part in parts]
        bad = [r for r in results if r.verdict == "contradicted"]
        if bad:
            return SemanticResult(
                "contradicted",
                [x for r in bad for x in r.reasons],
                bad[0].window,
                bad[0].nli,
                min(r.match_overlap for r in results),
            )
        disputed = [r for r in results if r.verdict == "disputed"]
        if disputed:
            return SemanticResult(
                "disputed",
                [x for r in disputed for x in r.reasons],
                disputed[0].window,
                disputed[0].nli,
                min(r.match_overlap for r in results),
            )
        weak = [r for r in results if r.verdict != "supported"]
        if weak:
            specific = any(r.verdict == "unverified" for r in weak)
            return SemanticResult(
                "unverified" if specific else "unconfirmed",
                [x for r in weak for x in r.reasons],
                weak[0].window,
                weak[0].nli,
                min(r.match_overlap for r in results),
            )
        top = results[0]
        return SemanticResult(
            "supported",
            [x for r in results for x in r.reasons][:3],
            top.window,
            top.nli,
            min(r.match_overlap for r in results),
        )

    # ----------------------------------------------------------------- one claim sentence

    def _check_part(
        self, claim: str, sentences: list[str], text: str, trusted: str = ""
    ) -> SemanticResult:
        claim_terms = _content_terms(claim)
        if not claim_terms or not sentences:
            return SemanticResult("unverified", ["nothing to compare"])
        ranked = sorted(
            ((len(claim_terms & _content_terms(s)) / len(claim_terms), s) for s in sentences),
            key=lambda x: -x[0],
        )
        best_overlap, best = ranked[0]
        res = SemanticResult("unverified", [], best, None, round(best_overlap, 3))
        matched = best_overlap >= self.cfg.min_match_overlap
        contradictions: list[str] = []  # hard evidence
        disputes: list[str] = []  # soft signals
        flags: list[str] = []
        quoted_ok = False

        if self.cfg.use_rules:
            sig = signature_conflict(claim, text)
            if sig:
                contradictions.append(sig)
            ids = identifiers(claim)
            low = text.lower()
            missing = [i for i in ids if not identifier_present(i, low) and i not in trusted]
            if missing:
                flags.append("identifier not in cited text: " + ", ".join(missing[:3]))
            # a sentence that only introduces code which appears verbatim in the cited text
            spans = _BACKTICK_SPAN.findall(claim)
            rest = _content_terms(_BACKTICK_SPAN.sub(" ", claim))
            if spans and len(rest) <= 3 and all(_norm(sp) in _norm(text) for sp in spans):
                quoted_ok = True
            window = " ".join(s for _, s in ranked[:2])
            # numbers inside a token the user typed (host:port, a path) are theirs, not a documentation claim
            cn = _numbers(claim) - set(_ATTACHED_NUMBER.findall(trusted))
            if matched:
                wn = _numbers(window)
                if cn and wn and not cn <= wn:
                    contradictions.append(
                        f"number {sorted(cn - wn)[0]} differs from the documented {sorted(wn - cn)[:2] or sorted(wn)[:2]}"
                    )
                if (_neg_count(claim) > 0) != (_neg_count(best) > 0) and not missing:
                    disputes.append("negation polarity differs from the matching sentence")
            elif cn and not cn <= _numbers(text):
                flags.append("number not found in cited text")

        entails = False
        nli_contra: list[str] = []
        if self.nli is not None:
            windows = [s for _, s in ranked[: self.cfg.max_windows]]
            if len(windows) >= 2:
                windows.append(windows[0] + " " + windows[1])
            scores = self.nli.score([(w, claim) for w in windows])
            top = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
            res.nli, res.window = scores[top], windows[top]
            entails = scores[top]["entailment"] >= self.cfg.entail_threshold
            if not entails:  # entailment by any window outranks another window's "contradiction"
                for w, sc in zip(windows, scores, strict=True):
                    w_overlap = len(claim_terms & _content_terms(w)) / len(claim_terms)
                    if (
                        sc["contradiction"] >= self.cfg.contradict_threshold
                        and w_overlap >= self.cfg.min_match_overlap
                    ):
                        nli_contra.append(
                            f"NLI contradiction ({sc['contradiction']:.2f}) with a matching sentence"
                        )  # soft
                        res.window, res.nli = w, sc
                        break

        disputes += nli_contra
        if entails:  # a strong entailment outranks the crude negation heuristic
            disputes = [d for d in disputes if not d.startswith("negation")]
        if contradictions:
            res.verdict, res.reasons = "contradicted", contradictions
        elif disputes:
            res.verdict, res.reasons = "disputed", disputes
        elif quoted_ok and not flags:
            res.verdict, res.reasons = "supported", ["quotes code that appears in the cited text"]
        elif self.nli is not None:
            if entails and not flags:
                res.verdict, res.reasons = "supported", ["entailed by a cited sentence"]
            elif entails:
                res.verdict, res.reasons = "unverified", ["entailed, but " + "; ".join(flags)]
            elif flags:
                res.verdict, res.reasons = "unverified", flags
            else:
                res.verdict, res.reasons = "unconfirmed", ["no cited sentence entails the claim"]
        else:  # rules only: accept only a close sentence match with no flags
            if matched and best_overlap >= 0.8 and not flags:
                res.verdict, res.reasons = (
                    "supported",
                    ["close match to a cited sentence; no rule fired"],
                )
            elif flags:
                res.verdict, res.reasons = "unverified", flags
            else:
                res.verdict, res.reasons = "unconfirmed", ["no close cited sentence"]
        return res


class GroundedSupportJudge:
    """Lexical (or LLM) support judge plus the semantic guard.

    * contradicted by hard evidence (number / signature) -> ``contradicted`` (the pipeline removes the claim);
    * a soft conflict signal (NLI / negation)            -> ``partial`` and marked ``disputed`` (kept, flagged);
    * lexically supported but a specific fact of the claim is absent from the cited text -> downgraded to
      ``partial`` and marked ``unverified`` (visible in the claim, counted, caps confidence);
    * nothing found either way                           -> unchanged, marked ``unconfirmed``;
    * lexically unsupported            -> stays ``unsupported``.
    """

    def __init__(self, base: SupportJudge, guard: SemanticGuard) -> None:
        self.base, self.guard = base, guard

    @property
    def name(self) -> str:
        return f"{self.base.name}+semantic"

    supports_trusted = True

    def judge(self, claim: str, passages: Sequence[str], trusted: str = "") -> SupportVerdict:
        if getattr(self.base, "supports_trusted", False):
            base = self.base.judge(claim, passages, trusted=trusted)  # type: ignore[call-arg]
        else:
            base = self.base.judge(claim, passages)
        sem = self.guard.check(claim, passages, trusted)
        why = "; ".join(sem.reasons)[:240]
        if sem.verdict == "contradicted":
            return SupportVerdict("contradicted", 0.0, why, "contradicted")
        if sem.verdict == "disputed":
            label = "partial" if base.label == "supported" else base.label
            return SupportVerdict(label, min(base.score, 0.5), f"disputed: {why}", "disputed")
        if base.label == "unchecked":
            return SupportVerdict(base.label, base.score, base.reason, sem.verdict)
        if sem.verdict == "supported":
            return SupportVerdict(base.label, base.score, base.reason, "supported")
        if sem.verdict == "unconfirmed":  # neutral: no contradiction found, nothing confirmed
            return SupportVerdict(base.label, base.score, base.reason, "unconfirmed")
        label = "partial" if base.label == "supported" else base.label
        score = min(base.score, 0.5) if base.label == "supported" else base.score
        return SupportVerdict(label, score, f"unverified: {why}", "unverified")
