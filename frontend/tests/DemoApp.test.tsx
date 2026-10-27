import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import DemoApp from "@/components/DemoApp";
import { DEMO_EXAMPLES, DEMO_META, type DemoExample } from "@/lib/demo";
import { isDebugResponse } from "@/lib/api";
import { ABSTAINED, response } from "./fixtures";

const examples: DemoExample[] = [
  { id: "a", title: "PyTorch: devices", framework: "pytorch", question: "RuntimeError: devices", note: "Good one.", response: response() },
  { id: "b", title: "Out of scope", framework: null, question: "Kubernetes pod crash", note: "Correct refusal.", response: ABSTAINED },
];

afterEach(() => vi.restoreAllMocks());

describe("DemoApp (static recorded demo)", () => {
  it("says plainly that it is a recorded, read-only demo with no server and no key", () => {
    render(<DemoApp examples={examples} />);
    const note = screen.getByRole("note");
    expect(note).toHaveTextContent("Recorded demo: read-only, no server behind it");
    expect(note).toHaveTextContent("no API key exists in this page");
    expect(note).toHaveTextContent("including a refusal and a weak answer");
  });

  it("lists the examples with their outcome and shows the recorded answer when one is picked", async () => {
    render(<DemoApp examples={examples} />);
    expect(screen.getByRole("button", { name: /PyTorch: devices/ })).toHaveTextContent("Answered");
    expect(screen.getByRole("button", { name: /Out of scope/ })).toHaveTextContent("Declined to answer");
    expect(screen.queryByRole("region", { name: "Likely cause" })).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /PyTorch: devices/ }));
    expect(screen.getByText("RuntimeError: devices")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Likely cause" })).toHaveTextContent("The tensors are on different devices");
    expect(screen.getByLabelText("Source documents")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /Out of scope/ }));
    expect(screen.getByRole("status")).toHaveTextContent("Not enough documentation to answer");
  });

  it("makes no network request at all and offers no way to send feedback", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<DemoApp examples={examples} />);
    await userEvent.click(screen.getByRole("button", { name: /PyTorch: devices/ }));
    await userEvent.click(screen.getAllByRole("button", { name: /Show source E1/ })[0]);
    expect(screen.getByText(/Feedback is disabled in this read-only demo/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Helpful/ })).toBeNull();
    expect(screen.queryByRole("textbox")).toBeNull(); // no question box: custom text cannot be sent anywhere
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("handles an empty bundle without crashing", () => {
    render(<DemoApp examples={[]} />);
    expect(screen.getByText(/No recorded examples are bundled/)).toBeInTheDocument();
  });
});

describe("the bundled recording", () => {
  it("is well formed: every example is a valid API response and the demo includes honest negatives", () => {
    if (DEMO_EXAMPLES.length === 0) return; // placeholder build
    expect(new Set(DEMO_EXAMPLES.map((e) => e.id)).size).toBe(DEMO_EXAMPLES.length);
    for (const e of DEMO_EXAMPLES) {
      expect(isDebugResponse(e.response)).toBe(true);
      expect(e.question.length).toBeGreaterThan(5);
      expect(e.note.length).toBeGreaterThan(10);
      expect(e.response.request_id).toBe(`demo-${e.id}`);
    }
    const statuses = new Set(DEMO_EXAMPLES.map((e) => e.response.status));
    expect(statuses.has("abstained")).toBe(true); // a refusal is shown on purpose
    expect(DEMO_EXAMPLES.some((e) => e.response.confidence === "low")).toBe(true); // and a weak answer
    expect(DEMO_META.model).toMatch(/Gemini/);
  });

  it("contains no secret-looking strings", () => {
    const text = JSON.stringify({ DEMO_EXAMPLES, DEMO_META });
    expect(text).not.toMatch(/AIza[0-9A-Za-z_-]{20,}/);
    expect(text).not.toMatch(/\bsk-[A-Za-z0-9_-]{16,}/);
    expect(text).not.toMatch(/GEMINI_API_KEY\s*=\s*\S+/);
    expect(within(document.body).queryAllByText(/AIza/)).toHaveLength(0);
  });
});
