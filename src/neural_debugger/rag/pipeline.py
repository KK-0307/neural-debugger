"""The RAG pipeline: question -> retrieve -> select evidence -> (abstain?) -> generate -> validate -> answer.

Every stage degrades explicitly instead of guessing:

* nothing usable retrieved, or the evidence cannot cover the question  -> ``abstained`` (the model is not called);
* the LLM provider fails (not configured, auth, rate limit, timeout...) -> ``error`` with the retrieved sources;
* the reply is not valid JSON                                          -> one repair attempt, then ``error``;
* claims without a valid citation, or with unsupported citations       -> removed, ``partial``/``abstained``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .evidence import EvidenceConfig, select_evidence
from .parse import ModelAnswer, parse_model_output
from .prompt import PROMPT_VERSION, build_repair_request, build_request
from .providers.base import LLMProvider, LLMRequest, LLMResponse, ProviderError
from .question import prepare_question
from .retrieval import SectionRetriever
from .sufficiency import Sufficiency, SufficiencyConfig, TermStats, assess
from .support import LexicalSupportJudge, LLMSupportJudge, SupportJudge
from .types import (
    AnswerStatus,
    Citation,
    Claim,
    Confidence,
    Evidence,
    QuestionInfo,
    RAGAnswer,
    Timings,
    ValidationReport,
)
from .validate import Validated, validate_answer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    retrieve_k: int = 12
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    support: Literal["off", "lexical", "llm"] = "lexical"
    max_output_tokens: int = 1500
    repair_attempts: int = 1
    drop_unsupported: bool = True
    generation_timeout_s: float = 60.0
    min_question_chars: int = 3


class RAGPipeline:
    def __init__(
        self,
        retriever: SectionRetriever,
        provider: LLMProvider,
        config: PipelineConfig | None = None,
        stats: TermStats | None = None,
        judge: SupportJudge | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.config = config or PipelineConfig()
        self.stats = stats
        self.judge: SupportJudge | None
        if judge is not None:
            self.judge = judge
        elif self.config.support == "lexical":
            self.judge = LexicalSupportJudge()
        elif self.config.support == "llm":
            self.judge = LLMSupportJudge(provider)
        else:
            self.judge = None

    # ------------------------------------------------------------------ public

    def answer(self, question: str, framework: str | None = None) -> RAGAnswer:
        t0 = time.perf_counter()
        timings = Timings()
        if len(question.strip()) < self.config.min_question_chars:
            return self._finish(self._empty(question, "abstained", "empty_question", timings), t0)
        q = prepare_question(question)
        trace: dict[str, Any] = {
            "prompt_version": PROMPT_VERSION,
            "retrieval_query": q.retrieval_query,
            "framework_filter": framework,
            "question_injection_flags": list(q.injection_flags),
            "question_truncated": q.truncated,
        }

        t = time.perf_counter()
        try:
            sections = self.retriever.search(q.retrieval_query, self.config.retrieve_k, framework)
        except Exception as exc:  # retriever failures must not leak internals or crash the request
            logger.exception("retrieval failed")
            timings.retrieval_ms = (time.perf_counter() - t) * 1000
            trace["retrieval_error"] = type(exc).__name__
            return self._finish(
                self._empty(question, "error", "retrieval_failed", timings, trace), t0
            )
        timings.retrieval_ms = (time.perf_counter() - t) * 1000

        evidence, selection = select_evidence(sections, self.config.evidence)
        trace["retrieved"] = len(sections)
        trace["dropped_evidence"] = selection.dropped
        trace["evidence_flags"] = {e.id: list(e.flags) for e in evidence if e.flags}
        suff = assess(q, evidence, self.stats, self.config.sufficiency)
        trace["sufficiency"] = {"level": suff.level, "coverage": round(suff.coverage, 3)}

        if suff.level == "none":
            ans = self._abstain(question, q, evidence, suff, timings, trace)
            return self._finish(ans, t0)

        t = time.perf_counter()
        request = build_request(q, evidence, suff, self.config.max_output_tokens)
        request = LLMRequest(**{**request.__dict__, "timeout_s": self.config.generation_timeout_s})
        try:
            parsed, response, usage = self._generate(request, trace)
        except ProviderError as exc:
            timings.generation_ms = (time.perf_counter() - t) * 1000
            trace["provider_error"] = {"kind": exc.kind, "retryable": exc.retryable}
            logger.warning("llm provider error: %s", exc.kind)
            ans = self._empty(question, "error", f"llm_{exc.kind}", timings, trace, evidence)
            return self._finish(ans, t0)
        timings.generation_ms = (time.perf_counter() - t) * 1000
        if parsed is None:
            ans = self._empty(question, "error", "malformed_model_output", timings, trace, evidence)
            ans.provider, ans.model = self.provider.name, response.model if response else None
            return self._finish(ans, t0)

        t = time.perf_counter()
        if parsed.status == "insufficient_evidence":
            ans = self._abstain(
                question,
                q,
                evidence,
                suff,
                timings,
                trace,
                "model_reported_insufficient_evidence",
                missing=list(parsed.missing_information),
            )
        else:
            validated = validate_answer(parsed, evidence, self.judge, self.config.drop_unsupported)
            ans = self._assemble(question, q, evidence, suff, validated, timings, trace)
        timings.validation_ms = (time.perf_counter() - t) * 1000
        ans.provider, ans.model = self.provider.name, response.model if response else None
        ans.usage = usage
        return self._finish(ans, t0)

    # ------------------------------------------------------------- generation

    def _generate(
        self, request: LLMRequest, trace: dict[str, Any]
    ) -> tuple[ModelAnswer | None, LLMResponse | None, dict[str, int]]:
        usage: dict[str, int] = {}
        response = self.provider.generate(request)
        _add_usage(usage, response)
        result = parse_model_output(response.text)
        attempts = 0
        while result.answer is None and attempts < self.config.repair_attempts:
            attempts += 1
            trace["repair_attempts"] = attempts
            response = self.provider.generate(
                build_repair_request(request, response.text, result.error or "invalid reply")
            )
            _add_usage(usage, response)
            result = parse_model_output(response.text)
        if result.answer is None:
            trace["parse_error"] = result.error
        return result.answer, response, usage

    # ------------------------------------------------------------- assembly

    def _assemble(
        self,
        question: str,
        q: QuestionInfo,
        evidence: list[Evidence],
        suff: Sufficiency,
        v: Validated,
        timings: Timings,
        trace: dict[str, Any],
    ) -> RAGAnswer:
        caveats = list(v.caveats)
        by_fw: dict[str, set[str]] = {}
        for e in evidence:
            by_fw.setdefault(e.framework, set()).add(e.version)
        for fw, ver in q.versions:
            doc = by_fw.get(fw)
            if doc and all(ver.split(".")[:2] != d.split(".")[:2] for d in doc):
                note = (
                    f"You mentioned {fw} {ver}; this answer is based on the indexed {fw} "
                    f"{'/'.join(sorted(doc))} documentation and behaviour may differ."
                )
                if note not in caveats:
                    caveats.append(note)
        for e in evidence:
            for n in e.version_notes:
                note = f"Version note in {e.title}: {n}"
                if e.id in {c for cl in _claims(v) for c in cl.citations} and note not in caveats:
                    caveats.append(note)
        if suff.level == "weak":
            caveats.append(f"Limited documentation coverage: {suff.reason}.")
        if len(by_fw) > 1:
            caveats.append(
                "The cited documentation spans several frameworks ("
                + ", ".join(sorted(by_fw))
                + "); check which one your code uses."
            )
        used = _claims(v)
        if not used:
            ans = self._abstain(
                question,
                q,
                evidence,
                suff,
                timings,
                trace,
                "no_supported_claims",
                report=v.report,
                missing=v.missing,
            )
            return ans
        citations = _citations(used, evidence)
        status: AnswerStatus = (
            "answered" if v.report.dropped_claims + v.report.unsupported_claims == 0 else "partial"
        )
        return RAGAnswer(
            status=status,
            question=question,
            diagnosis=v.diagnosis,
            steps=v.steps,
            code_suggestions=v.code,
            conflicts=v.conflicts,
            caveats=caveats,
            missing_information=v.missing,
            citations=citations,
            confidence=self._confidence(suff, v),
            abstain_reason=None,
            evidence=evidence,
            validation=v.report,
            timings=timings,
            provider=self.provider.name,
            model=None,
            trace=trace,
        )

    def _confidence(self, suff: Sufficiency, v: Validated) -> Confidence:
        r = v.report
        if r.invalid_citations or r.unsupported_claims or r.dropped_claims > r.total_claims / 2:
            return "low"
        if suff.level == "weak" or r.partially_supported_claims or r.dropped_claims or v.conflicts:
            return "medium" if self.judge is not None else "low"
        return "high" if self.judge is not None else "medium"

    def _abstain(
        self,
        question: str,
        q: QuestionInfo,
        evidence: list[Evidence],
        suff: Sufficiency,
        timings: Timings,
        trace: dict[str, Any],
        reason: str = "insufficient_evidence",
        report: ValidationReport | None = None,
        missing: list[str] | None = None,
    ) -> RAGAnswer:
        miss = list(missing or [])
        if not miss and suff.missing_terms:
            miss.append("No documentation found for: " + ", ".join(suff.missing_terms[:6]) + ".")
        miss.append(
            "Try including the exact error message, the function or class involved, and which library "
            "(Python, PyTorch or FastAPI) you are using."
        )
        return RAGAnswer(
            status="abstained",
            question=question,
            diagnosis=None,
            steps=[],
            code_suggestions=[],
            conflicts=[],
            caveats=[suff.reason] if suff.reason else [],
            missing_information=miss,
            citations=[],
            confidence="none",
            abstain_reason=reason,
            evidence=evidence,
            validation=report or ValidationReport(),
            timings=timings,
            provider=None if reason == "insufficient_evidence" else self.provider.name,
            model=None,
            trace=trace,
        )

    def _empty(
        self,
        question: str,
        status: AnswerStatus,
        reason: str,
        timings: Timings,
        trace: dict[str, Any] | None = None,
        evidence: list[Evidence] | None = None,
    ) -> RAGAnswer:
        return RAGAnswer(
            status=status,
            question=question,
            diagnosis=None,
            steps=[],
            code_suggestions=[],
            conflicts=[],
            caveats=[],
            missing_information=[],
            citations=[],
            confidence="none",
            abstain_reason=reason,
            evidence=evidence or [],
            validation=ValidationReport(),
            timings=timings,
            provider=self.provider.name,
            model=None,
            trace=trace or {},
        )

    @staticmethod
    def _finish(ans: RAGAnswer, t0: float) -> RAGAnswer:
        ans.timings.total_ms = (time.perf_counter() - t0) * 1000
        return ans


def _add_usage(total: dict[str, int], resp: LLMResponse) -> None:
    for k, v in resp.usage.items():
        total[k] = total.get(k, 0) + v


def _claims(v: Validated) -> list[Claim]:
    out: list[Claim] = []
    if v.diagnosis:
        out.append(v.diagnosis)
    out += v.steps
    out += [c.explanation for c in v.code]
    out += v.conflicts
    return out


def _citations(claims: list[Claim], evidence: list[Evidence]) -> list[Citation]:
    by_id = {e.id: e for e in evidence}
    seen: list[str] = []
    for c in claims:
        for cid in c.citations:
            if cid not in seen:
                seen.append(cid)
    return [
        Citation(e.id, e.section_id, e.title, e.heading, e.url, e.framework, e.version)
        for e in (by_id[i] for i in seen)
    ]
