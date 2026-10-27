/** Mirrors the backend's `POST /api/v1/debug` and `POST /api/v1/feedback` payloads (src/neural_debugger/backend/api). */

export type AnswerStatus = "answered" | "partial" | "abstained" | "error";
export type Confidence = "high" | "medium" | "low" | "none";
export type Framework = "python" | "pytorch" | "fastapi";

/** lexical support of a claim by the text it cites */
export type SupportLabel = "supported" | "partial" | "unsupported" | "contradicted" | "unchecked";
/** semantic grounding: `disputed` = the cited text may say the opposite, `unverified` = a name/number is not in it */
export type SemanticLabel = "supported" | "unverified" | "disputed" | "unconfirmed" | "unchecked";

export type Claim = {
  text: string;
  citations: string[];
  support: SupportLabel;
  support_score: number | null;
  semantic: SemanticLabel;
  semantic_reason: string;
};

export type CodeSuggestion = { language: string; code: string; explanation: Claim };

export type Citation = {
  id: string;
  title: string;
  heading: string;
  url: string;
  framework: string;
  version: string;
};

export type Source = Citation & { cited: boolean; flags: string[]; excerpt: string };

export type ValidationSummary = {
  total_claims: number;
  dropped_claims: number;
  unsupported_claims: number;
  partially_supported_claims: number;
  contradicted_claims: number;
  disputed_claims: number;
  unverified_claims: number;
  unconfirmed_claims: number;
  invalid_citations: string[];
  stripped_urls: number;
};

export type DebugResponse = {
  request_id: string | null;
  status: AnswerStatus;
  confidence: Confidence;
  diagnosis: Claim | null;
  steps: Claim[];
  code_suggestions: CodeSuggestion[];
  conflicts: Claim[];
  caveats: string[];
  missing_information: string[];
  citations: Citation[];
  sources: Source[];
  abstain_reason: string | null;
  validation: ValidationSummary;
  timings: { retrieval_ms: number; generation_ms: number; validation_ms: number; total_ms: number };
  provider: string | null;
  model: string | null;
  evidence_gate: { level: "none" | "weak" | "ok"; coverage: number } | null;
};

export type FeedbackReason =
  | "wrong_answer"
  | "incomplete"
  | "wrong_source"
  | "unclear"
  | "should_have_answered"
  | "should_have_abstained"
  | "other";

export type FeedbackInput = {
  request_id: string;
  rating: "helpful" | "not_helpful";
  reasons: FeedbackReason[];
  comment: string;
  answer_status: AnswerStatus;
  confidence: Confidence;
  include_question: boolean;
  question?: string;
};
