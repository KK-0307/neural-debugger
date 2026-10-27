import { describe, expect, it } from "vitest";
import { claimCitationIds, safeHref, splitCitations } from "@/lib/citations";
import { addEntry, MAX_ENTRIES, parseHistory, type HistoryEntry } from "@/lib/history";
import { claimMark, explainConfidence, provenanceNote, visibleCaveats } from "@/lib/uncertainty";
import { ABSTAINED, claim, response } from "./fixtures";

describe("citations", () => {
  it("splits inline [E#] markers (single and grouped) out of plain text", () => {
    expect(splitCitations("Use `to()` [E1] or `cuda()` [e2, E3].")).toEqual([
      { kind: "text", text: "Use `to()` " },
      { kind: "cite", ids: ["E1"] },
      { kind: "text", text: " or `cuda()` " },
      { kind: "cite", ids: ["E2", "E3"] },
      { kind: "text", text: "." },
    ]);
    expect(splitCitations("no markers [see docs] here")).toEqual([{ kind: "text", text: "no markers [see docs] here" }]);
  });

  it("merges structured and inline ids without duplicates", () => {
    expect(claimCitationIds("Do X [E2] and Y [E4].", ["E1", "E2"])).toEqual(["E1", "E2", "E4"]);
  });

  it("only ever links http(s) URLs", () => {
    expect(safeHref("https://docs.python.org/3.12/")).toBe("https://docs.python.org/3.12/");
    expect(safeHref("javascript:alert(1)")).toBeNull();
    expect(safeHref("data:text/html,<script>1</script>")).toBeNull();
    expect(safeHref("not a url")).toBeNull();
  });
});

describe("explainConfidence", () => {
  it("restates only what the validators reported", () => {
    const e = explainConfidence(
      response({
        confidence: "low",
        status: "partial",
        validation: {
          ...response().validation,
          dropped_claims: 1,
          unsupported_claims: 2,
          contradicted_claims: 1,
          disputed_claims: 1,
          unverified_claims: 1,
          partially_supported_claims: 3,
          invalid_citations: ["E9"],
        },
        caveats: ["You mentioned pytorch 1.9; this answer is based on the indexed pytorch 2.14 documentation."],
        evidence_gate: { level: "weak", coverage: 0.47 },
      }),
    );
    const text = e.reasons.map((r) => r.text).join("\n");
    expect(e.headline).toBe("Low confidence");
    expect(text).toMatch(/covers only 47%/);
    expect(text).toMatch(/1 statement was removed because it had no valid citation/);
    expect(text).toMatch(/2 statements were removed because the cited text did not support/);
    expect(text).toMatch(/contradicted the cited text/);
    expect(text).toMatch(/1 statement is flagged/);
    expect(text).toMatch(/E9, which it was never shown/);
    expect(text).toMatch(/version that differs/);
  });

  it("reports a clean high-confidence answer without inventing warnings", () => {
    const e = explainConfidence(response({ confidence: "high", evidence_gate: { level: "ok", coverage: 0.8 } }));
    expect(e.tone).toBe("good");
    expect(e.reasons.filter((r) => r.tone === "warn" || r.tone === "bad")).toEqual([]);
  });

  it("says plainly that an abstention is a refusal to guess", () => {
    const e = explainConfidence(ABSTAINED);
    expect(e.headline).toBe("No confident answer");
    expect(e.reasons.map((r) => r.text).join(" ")).toMatch(/rather than risk an unsupported one/);
  });
});

describe("claimMark", () => {
  it("flags disputed and unverified statements and stays quiet about neutral ones", () => {
    expect(claimMark(claim("x", [], { semantic: "disputed", semantic_reason: "NLI contradiction" }))?.label).toBe("Check this");
    expect(claimMark(claim("x", [], { semantic: "unverified" }))?.label).toBe("Not in cited text");
    expect(claimMark(claim("x", [], { support: "partial", semantic: "unconfirmed" }))?.label).toBe("Partly supported");
    expect(claimMark(claim("x", [], { support: "unchecked", semantic: "unconfirmed" }))).toBeNull();
  });
});

describe("history", () => {
  const entry = (q: string): HistoryEntry => ({ id: q, at: 1, question: q, framework: null, result: response() });

  it("keeps the newest first, de-duplicates by question and caps the length", () => {
    let h: HistoryEntry[] = [];
    for (let i = 0; i < MAX_ENTRIES + 3; i++) h = addEntry(h, entry(`q${i}`));
    expect(h).toHaveLength(MAX_ENTRIES);
    expect(h[0].question).toBe(`q${MAX_ENTRIES + 2}`);
    h = addEntry(h, entry("q5"));
    expect(h[0].question).toBe("q5");
    expect(h.filter((e) => e.question === "q5")).toHaveLength(1);
  });

  it("survives corrupt stored data", () => {
    expect(parseHistory("not json")).toEqual([]);
    expect(parseHistory('{"a":1}')).toEqual([]);
    expect(parseHistory('[{"question":5}]')).toEqual([]);
    expect(parseHistory(null)).toEqual([]);
  });
});

describe("provenance and caveats", () => {
  it("never claims a language model wrote an answer that no model produced", () => {
    expect(provenanceNote(ABSTAINED)).toMatch(/No language model was called/);
    expect(provenanceNote(response({ provider: "extractive", model: "extractive-v1" }))).toMatch(/offline test provider/);
    expect(provenanceNote(response())).toMatch(/Written by a language model/);
    expect(provenanceNote(response())).toMatch(/no human reviewed it/);
  });

  it("does not repeat coverage caveats that the confidence panel already shows", () => {
    const r = response({ caveats: ["Limited documentation coverage: partial coverage (59%).", "retrieved documentation covers only 21% of the question's key terms", "Version note in typing: Added in version 3.11."] });
    expect(visibleCaveats(r)).toEqual(["Version note in typing: Added in version 3.11."]);
  });
});
