import type { Claim, Confidence, DebugResponse } from "./types";

export type Tone = "good" | "info" | "warn" | "bad";
export type Reason = { tone: Tone; text: string };

export type Explanation = {
  headline: string;
  summary: string;
  tone: Tone;
  reasons: Reason[];
};

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
const pronoun = (n: number) => (n === 1 ? "it" : "them");

const HEADLINES: Record<Confidence, { headline: string; tone: Tone }> = {
  high: { headline: "High confidence", tone: "good" },
  medium: { headline: "Medium confidence", tone: "info" },
  low: { headline: "Low confidence", tone: "warn" },
  none: { headline: "No confident answer", tone: "bad" },
};

/** All statements of an answer that carry a claim object (diagnosis, steps, code explanations, conflicts). */
export function allClaims(r: DebugResponse): Claim[] {
  return [
    ...(r.diagnosis ? [r.diagnosis] : []),
    ...r.steps,
    ...r.code_suggestions.map((c) => c.explanation),
    ...r.conflicts,
  ];
}

/**
 * Turn the validators' counters into sentences a developer can act on. Nothing here is a new judgement: every line
 * restates a number the backend reported, so the UI cannot claim more certainty than the pipeline measured.
 */
export function explainConfidence(r: DebugResponse): Explanation {
  const base = HEADLINES[r.confidence];
  const v = r.validation;
  const reasons: Reason[] = [];

  if (r.evidence_gate) {
    const pct = Math.round(r.evidence_gate.coverage * 100);
    if (r.evidence_gate.level === "ok") {
      reasons.push({ tone: "good", text: `The retrieved documentation covers ${pct}% of the key terms in your question.` });
    } else if (r.evidence_gate.level === "weak") {
      reasons.push({ tone: "warn", text: `The retrieved documentation covers only ${pct}% of the key terms in your question.` });
    } else {
      reasons.push({ tone: "bad", text: `The retrieved documentation covers only ${pct}% of the key terms in your question.` });
    }
  }
  if (v.dropped_claims > 0) {
    reasons.push({ tone: "warn", text: `${plural(v.dropped_claims, "statement was", "statements were")} removed because ${pronoun(v.dropped_claims) === "it" ? "it" : "they"} had no valid citation.` });
  }
  if (v.unsupported_claims > 0) {
    reasons.push({ tone: "warn", text: `${plural(v.unsupported_claims, "statement was", "statements were")} removed because the cited text did not support ${pronoun(v.unsupported_claims)}.` });
  }
  if (v.contradicted_claims > 0) {
    reasons.push({ tone: "warn", text: `${plural(v.contradicted_claims, "statement was", "statements were")} removed because a number or API signature contradicted the cited text.` });
  }
  if (v.disputed_claims > 0) {
    reasons.push({ tone: "warn", text: `${plural(v.disputed_claims, "statement is", "statements are")} flagged: the cited text may say the opposite. Check the source.` });
  }
  if (v.unverified_claims > 0) {
    reasons.push({ tone: "info", text: `${plural(v.unverified_claims, "statement mentions", "statements mention")} a name or number that does not appear in the cited text.` });
  }
  if (v.partially_supported_claims > 0) {
    reasons.push({ tone: "info", text: `${plural(v.partially_supported_claims, "statement is", "statements are")} only partly backed by the cited text.` });
  }
  if (v.invalid_citations.length > 0) {
    reasons.push({ tone: "warn", text: `The model cited ${v.invalid_citations.join(", ")}, which it was never shown; those citations were discarded.` });
  }
  if (r.conflicts.length > 0) {
    reasons.push({ tone: "info", text: "The documentation appears to disagree with itself or across versions; see the conflicts section." });
  }
  if (r.caveats.some((c) => /^You mentioned /.test(c))) {
    reasons.push({ tone: "warn", text: "You mentioned a version that differs from the documentation that was indexed; behaviour may differ." });
  }
  if (r.status === "abstained") {
    reasons.push({ tone: "bad", text: "No answer was given rather than risk an unsupported one." });
  }

  const summary =
    r.status === "answered"
      ? "Every statement below cites documentation that was retrieved for this question."
      : r.status === "partial"
        ? "Some statements were removed or flagged during validation; what remains cites retrieved documentation."
        : r.status === "abstained"
          ? "The retrieved documentation did not contain enough to answer this safely."
          : "No answer could be produced.";
  return { headline: base.headline, tone: base.tone, summary, reasons };
}

/** Who wrote this, said truthfully: a real model, the offline test provider, or nobody (declined before generation). */
export function provenanceNote(r: DebugResponse): string {
  if (r.status === "abstained" && !r.provider) {
    return "No language model was called: the question was declined before generation because the retrieved documentation did not cover it. Nothing here was run, and no human reviewed it.";
  }
  if (r.provider === "extractive" || r.model?.startsWith("extractive")) {
    return "Assembled from documentation sentences by the offline test provider. No language model wrote this, so expect stilted text. Citations are still checked automatically.";
  }
  return "Written by a language model from the retrieved documentation, then checked automatically (citations exist, the cited text backs each statement). Nothing here was run, and no human reviewed it. Verify before you rely on it.";
}

/** Coverage caveats are already stated in "How sure is this?"; do not repeat them under "Keep in mind". */
export function visibleCaveats(r: DebugResponse): string[] {
  return r.caveats.filter((c) => !/^(Limited documentation coverage|retrieved documentation covers)/i.test(c));
}

export type ClaimMark = { label: string; tone: Tone; hint: string } | null;

/** The small per-statement badge. "Neutral" outcomes (nothing contradicts it, nothing confirms it) show nothing. */
export function claimMark(c: Claim): ClaimMark {
  if (c.semantic === "disputed") {
    return { label: "Check this", tone: "warn", hint: c.semantic_reason || "The cited text may say the opposite." };
  }
  if (c.semantic === "unverified") {
    return { label: "Not in cited text", tone: "info", hint: c.semantic_reason || "A name or number in this statement is not in the cited text." };
  }
  if (c.support === "partial") {
    return { label: "Partly supported", tone: "info", hint: "Only part of this statement is backed by the cited text." };
  }
  if (c.support === "supported" || c.semantic === "supported") {
    return { label: "Supported", tone: "good", hint: "The cited text backs this statement (an automated check, not a human review)." };
  }
  return null;
}
