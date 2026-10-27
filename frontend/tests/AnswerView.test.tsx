import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import AnswerView from "@/components/AnswerView";
import { ABSTAINED, claim, response, source } from "./fixtures";

describe("AnswerView", () => {
  it("shows the diagnosis, numbered fixes, example code and the status banner", () => {
    render(<AnswerView result={response()} question="q" />);
    expect(screen.getByRole("status")).toHaveTextContent("Answered");
    expect(screen.getByRole("status")).toHaveTextContent("Medium confidence");
    const diagnosis = screen.getByRole("region", { name: "Likely cause" });
    expect(diagnosis).toHaveTextContent("The tensors are on different devices");
    const fixes = screen.getByRole("region", { name: "Suggested fixes" });
    expect(within(fixes).getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByText(/x = x\.to\(device\)/)).toBeInTheDocument();
    expect(screen.getByText(/illustrative · not run by Neural Debugger/)).toBeInTheDocument();
  });

  it("turns inline [E1] markers and listed citations into chips that open the matching source", async () => {
    render(<AnswerView result={response()} question="q" />);
    const diagnosis = screen.getByRole("region", { name: "Likely cause" });
    expect(diagnosis).not.toHaveTextContent("[E1]"); // the marker became a chip
    await userEvent.click(within(diagnosis).getByRole("button", { name: "Show source E1" }));
    const card = document.getElementById("source-E1") as HTMLElement;
    expect(card.className).toMatch(/border-indigo/);
    expect((card.querySelector("details") as HTMLDetailsElement).open).toBe(true);
    expect(within(card).getByText("Documentation text of E1.")).toBeInTheDocument();
  });

  it("lists cited sources first with safe external links, and uncited ones separately", () => {
    render(
      <AnswerView
        result={response({ sources: [source("E1"), source("E2", { url: "javascript:alert(1)" }), source("E3", { cited: false })] })}
        question="q"
      />,
    );
    const aside = screen.getByLabelText("Source documents");
    expect(within(aside).getByText("Cited in the answer (2)")).toBeInTheDocument();
    expect(within(aside).getByText("Also retrieved, not cited (1)")).toBeInTheDocument();
    const links = within(aside).getAllByRole("link", { name: /Open in the official documentation/ });
    expect(links).toHaveLength(2); // E2's javascript: URL is not rendered as a link
    for (const l of links) {
      expect(l).toHaveAttribute("target", "_blank");
      expect(l).toHaveAttribute("rel", "noopener noreferrer");
    }
    expect(within(aside).getByText("No safe link available.")).toBeInTheDocument();
  });

  it("makes statements that need attention visible and explains why confidence is lower", () => {
    const r = response({
      status: "partial",
      confidence: "low",
      steps: [
        claim("Set `weights_only=False`.", ["E1"], { semantic: "disputed", semantic_reason: "NLI contradiction (0.93)" }),
        claim("Use `torch.cuda.fancy()`.", ["E2"], { support: "partial", semantic: "unverified", semantic_reason: "identifier not in cited text" }),
      ],
      validation: { ...response().validation, unsupported_claims: 2, disputed_claims: 1, unverified_claims: 1 },
    });
    render(<AnswerView result={r} question="q" />);
    expect(screen.getByRole("status")).toHaveTextContent("Partly answered");
    expect(screen.getByText("Check this")).toHaveAttribute("title", "NLI contradiction (0.93)");
    expect(screen.getByText("Not in cited text")).toBeInTheDocument();
    const panel = screen.getByRole("region", { name: "How sure is this?" });
    expect(panel).toHaveTextContent("2 statements were removed because the cited text did not support them");
    expect(panel).toHaveTextContent("no human reviewed it");
  });

  it("explains an abstention instead of showing an empty answer, and offers the closest documentation", () => {
    render(<AnswerView result={ABSTAINED} question="TensorFlow problem" />);
    expect(screen.getByRole("status")).toHaveTextContent("Not enough documentation to answer");
    expect(screen.queryByRole("region", { name: "Likely cause" })).toBeNull();
    expect(screen.getByRole("region", { name: "What the documentation did not cover" })).toHaveTextContent("tensorflow, keras");
    expect(screen.getByText("Closest documentation found")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "How sure is this?" })).toHaveTextContent("rather than risk an unsupported one");
  });

  it("renders model output as inert text, never as HTML", () => {
    const evil = '<img src=x onerror="alert(1)"> <script>alert(2)</script>';
    render(<AnswerView result={response({ diagnosis: claim(evil, []) })} question="q" />);
    const region = screen.getByRole("region", { name: "Likely cause" });
    expect(region).toHaveTextContent(evil);
    expect(region.querySelector("img, script")).toBeNull();
  });

  it("collapses long caveat lists", async () => {
    const caveats = Array.from({ length: 6 }, (_, i) => `Version note ${i}`);
    render(<AnswerView result={response({ caveats })} question="q" />);
    expect(screen.queryByText("Version note 5")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Show 3 more" }));
    expect(screen.getByText("Version note 5")).toBeInTheDocument();
  });
});
