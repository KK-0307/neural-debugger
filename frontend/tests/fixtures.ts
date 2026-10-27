import type { Claim, DebugResponse, Source } from "@/lib/types";

export function claim(text: string, citations: string[] = ["E1"], over: Partial<Claim> = {}): Claim {
  return { text, citations, support: "supported", support_score: 1, semantic: "unconfirmed", semantic_reason: "", ...over };
}

export function source(id: string, over: Partial<Source> = {}): Source {
  return {
    id,
    title: `Page ${id}`,
    heading: `Page ${id} > Section`,
    url: `https://docs.example.org/${id}`,
    framework: "pytorch",
    version: "2.14",
    cited: true,
    flags: [],
    excerpt: `Documentation text of ${id}.`,
    ...over,
  };
}

const VALIDATION = {
  total_claims: 3,
  dropped_claims: 0,
  unsupported_claims: 0,
  partially_supported_claims: 0,
  contradicted_claims: 0,
  disputed_claims: 0,
  unverified_claims: 0,
  unconfirmed_claims: 0,
  invalid_citations: [],
  stripped_urls: 0,
};

export function response(over: Partial<DebugResponse> = {}): DebugResponse {
  return {
    request_id: "req-123",
    status: "answered",
    confidence: "medium",
    diagnosis: claim("The tensors are on different devices, one on cuda:0 and one on cpu. [E1]", ["E1"]),
    steps: [claim("Move both tensors with `tensor.to(device)`.", ["E1", "E2"]), claim("Call `model.to(device)` once.", ["E2"])],
    code_suggestions: [
      { language: "python", code: "x = x.to(device)\nmodel = model.to(device)", explanation: claim("Move the data and the module.", ["E2"]) },
    ],
    conflicts: [],
    caveats: ["Limited documentation coverage: partial coverage (59%) of the key terms."],
    missing_information: [],
    citations: [],
    sources: [source("E1"), source("E2", { framework: "pytorch" }), source("E3", { cited: false })],
    abstain_reason: null,
    validation: VALIDATION,
    timings: { retrieval_ms: 12, generation_ms: 4200, validation_ms: 3, total_ms: 4300 },
    provider: "openai_compatible",
    model: "gemini-3.1-flash-lite",
    evidence_gate: { level: "weak", coverage: 0.59 },
    ...over,
  };
}

export const ABSTAINED = response({
  status: "abstained",
  confidence: "none",
  diagnosis: null,
  steps: [],
  code_suggestions: [],
  caveats: [],
  missing_information: ["No documentation found for: tensorflow, keras.", "Try including the exact error message."],
  abstain_reason: "insufficient_evidence",
  provider: null, // a pre-LLM abstention: no model was called
  model: null,
  sources: [source("E1", { cited: false })],
  evidence_gate: { level: "none", coverage: 0.2 },
  validation: { ...VALIDATION, total_claims: 0 },
});
