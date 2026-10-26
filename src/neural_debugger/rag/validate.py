"""Validate the model's answer against the evidence it was shown, and clean what is returned.

Citation **validity** (this module): every citation must name an ``Evidence`` id that was in the prompt;
claims with no valid citation are removed; inline ``[E#]`` markers are checked the same way; URLs that are
not source URLs, markdown images and HTML tags are stripped from prose. Citation **support** is checked
separately by a ``SupportJudge``. Code is returned as inert text and is never executed.
"""

from __future__ import annotations

import re
from urllib.parse import urldefrag

from .parse import ModelAnswer
from .support import SupportJudge
from .text import term_set
from .types import Claim, CodeSuggestion, Evidence, ValidationReport

_INLINE = re.compile(r"\[\s*(E\d{1,3}(?:\s*,\s*E\d{1,3})*)\s*\]", re.I)
_URL = re.compile(r"https?://[^\s<>)\]\"']+", re.I)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_TAG = re.compile(r"</?[A-Za-z][^>\n]{0,200}>")
_RISKY_CODE = (
    ("rm -rf", re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", re.I)),
    ("pipe-to-shell download", re.compile(r"(curl|wget)[^\n|]*\|\s*(sudo\s+)?(ba)?sh", re.I)),
    ("sudo", re.compile(r"\bsudo\b")),
    ("os.system/subprocess", re.compile(r"\bos\.system\(|\bsubprocess\.|\bPopen\(")),
    ("eval/exec", re.compile(r"(?<![\w.])(eval|exec)\(")),
)


class Validated:
    def __init__(self) -> None:
        self.diagnosis: Claim | None = None
        self.steps: list[Claim] = []
        self.code: list[CodeSuggestion] = []
        self.conflicts: list[Claim] = []
        self.caveats: list[str] = []
        self.missing: list[str] = []
        self.report = ValidationReport()

    @property
    def claim_count(self) -> int:
        return (1 if self.diagnosis else 0) + len(self.steps) + len(self.code) + len(self.conflicts)


def _norm_url(url: str) -> str:
    return urldefrag(url.rstrip(".,;:"))[0].rstrip("/").lower()


class _Cleaner:
    def __init__(self, evidence: list[Evidence], report: ValidationReport) -> None:
        self.ids = {e.id for e in evidence}
        self.allowed = {_norm_url(e.url) for e in evidence}
        self.report = report

    def prose(self, text: str) -> str:
        def url(m: re.Match[str]) -> str:
            if _norm_url(m.group(0)) in self.allowed:
                return m.group(0)
            self.report.stripped_urls.append(m.group(0)[:100])
            return "[link removed]"

        n_before = len(text)
        text = _IMAGE.sub("", text)
        text = _TAG.sub("", text)
        if len(text) != n_before:
            self.report.stripped_markup += 1
        return _URL.sub(url, text).strip()

    def split_inline(self, text: str, cited: list[str]) -> tuple[str, list[str]]:
        """Keep valid inline ``[E#]`` markers (adding them to the citation list), drop invalid ones."""
        ids = list(cited)

        def repl(m: re.Match[str]) -> str:
            good = []
            for cid in re.findall(r"E\d{1,3}", m.group(1), re.I):
                cid = cid.upper()
                if cid in self.ids:
                    good.append(cid)
                    if cid not in ids:
                        ids.append(cid)
                else:
                    self.report.invalid_citations.append(cid)
                    self.report.citations_checked += 1
            return f"[{', '.join(good)}]" if good else ""

        return _INLINE.sub(repl, text).strip(), ids

    def valid(self, cited: list[str]) -> list[str]:
        good = []
        self.report.citations_checked += len(cited)
        for c in cited:
            if c in self.ids:
                if c not in good:
                    good.append(c)
            else:
                self.report.invalid_citations.append(c)
        return good


def validate_answer(
    model: ModelAnswer,
    evidence: list[Evidence],
    judge: SupportJudge | None = None,
    drop_unsupported: bool = True,
) -> Validated:
    out = Validated()
    rep = out.report
    cleaner = _Cleaner(evidence, rep)
    by_id = {e.id: e for e in evidence}

    def check(
        text: str, cites: list[str], kind: str, min_cites: int = 1, judge_text: str | None = None
    ) -> Claim | None:
        rep.total_claims += 1
        text = cleaner.prose(text)
        text, cites = cleaner.split_inline(text, cites)
        good = cleaner.valid(cites)
        if not text or len(good) < min_cites:
            rep.dropped_claims += 1
            return None
        label, score = "unchecked", None
        if judge is not None:
            passages = [f"{by_id[c].title} > {by_id[c].heading}\n{by_id[c].text}" for c in good]
            verdict = judge.judge(judge_text or text, passages)
            label, score = verdict.label, verdict.score
            if label == "unsupported":
                rep.unsupported_claims += 1
                if drop_unsupported:
                    return None
            elif label == "partial":
                rep.partially_supported_claims += 1
        return Claim(text, tuple(good), kind, label, score)  # type: ignore[arg-type]

    if model.diagnosis:
        out.diagnosis = check(model.diagnosis.text, model.diagnosis.citations, "diagnosis")
    for s in model.steps:
        c = check(s.text, s.citations, "step")
        if c:
            out.steps.append(c)
    for item in model.code_suggestions:
        explanation = item.explanation or "Suggested code."
        # a one-line "example from the docs" explanation cannot be checked; judge the code itself then
        judge_text = explanation if len(term_set(explanation)) >= 4 else item.code[:800]
        expl = check(explanation, item.citations, "code", judge_text=judge_text)
        if expl:
            out.code.append(CodeSuggestion(item.language.strip()[:20] or "text", item.code, expl))
            evidence_text = "\n".join(e.text for e in evidence)
            risky = [
                n
                for n, pat in _RISKY_CODE
                if pat.search(item.code) and not pat.search(evidence_text)
            ]
            if risky:
                out.caveats.append(
                    "A suggested snippet uses "
                    + ", ".join(risky)
                    + ", which does not appear in the "
                    "cited documentation. Review it carefully; it has not been run."
                )
    for cf in model.conflicts:
        c = check(cf.text, cf.citations, "conflict", min_cites=2)
        if c:
            out.conflicts.append(c)
    out.caveats += [cleaner.prose(t) for t in model.caveats if cleaner.prose(t)]
    out.missing = [cleaner.prose(t) for t in model.missing_information if cleaner.prose(t)]
    rep.invalid_citations = sorted(set(rep.invalid_citations))
    return out
