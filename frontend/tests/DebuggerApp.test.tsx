import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import DebuggerApp from "@/components/DebuggerApp";
import { startApi, type Router } from "./api-server";
import { ABSTAINED, response } from "./fixtures";

let close: (() => Promise<void>) | undefined;
beforeEach(() => window.localStorage.clear());
afterEach(async () => {
  await close?.();
  close = undefined;
});

async function setup(router: Router) {
  const api = await startApi(router);
  close = api.close;
  const user = userEvent.setup();
  render(<DebuggerApp apiBaseUrl={api.url} />);
  return { api, user };
}

const ok: Router = (req) =>
  req.path === "/api/v1/debug" ? { status: 200, body: response() } : { status: 201, body: { id: "f1", stored_question: false } };

async function ask(user: ReturnType<typeof userEvent.setup>, text = "RuntimeError: Expected all tensors to be on the same device") {
  await user.click(screen.getByLabelText(/Paste a stack trace/));
  await user.paste(text);
  await user.click(screen.getByRole("button", { name: "Diagnose" }));
}

describe("DebuggerApp", () => {
  it("cannot submit an empty or too-short question", async () => {
    await setup(ok);
    const btn = screen.getByRole("button", { name: "Diagnose" });
    expect(btn).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/Paste a stack trace/), "ab");
    expect(btn).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/Paste a stack trace/), "c");
    expect(btn).toBeEnabled();
  });

  it("sends the question and framework, shows progress, then the answer", async () => {
    const { api, user } = await setup(ok);
    await user.selectOptions(screen.getByLabelText("Documentation"), "pytorch");
    await ask(user);
    expect(await screen.findByRole("status", { name: "" })).toBeTruthy();
    expect(await screen.findByRole("region", { name: "Likely cause" })).toBeInTheDocument();
    const call = api.requests.find((r) => r.path === "/api/v1/debug");
    expect(call?.body).toEqual({ question: "RuntimeError: Expected all tensors to be on the same device", framework: "pytorch" });
  });

  it("submits with Ctrl+Enter and fills the example prompts", async () => {
    const { api, user } = await setup(ok);
    await user.click(screen.getByRole("button", { name: /PyTorch: tensors on different devices/ }));
    expect(screen.getByLabelText("Documentation")).toHaveValue("pytorch");
    await user.click(screen.getByLabelText(/Paste a stack trace/));
    await user.keyboard("{Control>}{Enter}{/Control}");
    await screen.findByRole("region", { name: "Likely cause" });
    expect(api.requests.filter((r) => r.path === "/api/v1/debug")).toHaveLength(1);
  });

  it("explains a server without a language model and still lists the closest documentation", async () => {
    const { user } = await setup(() => ({
      status: 503,
      body: {
        error: {
          code: "llm_not_configured",
          message: "No LLM provider is configured on this server.",
          details: [{ sources: [{ id: "E1", title: "CUDA semantics", url: "https://docs.example.org/cuda" }], reason: "llm_not_configured" }],
        },
      },
    }));
    await ask(user);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("No language model is configured on the server");
    expect(alert).toHaveTextContent("ND_LLM_PROVIDER");
    expect(within(alert).getByRole("link", { name: "CUDA semantics" })).toHaveAttribute("href", "https://docs.example.org/cuda");
  });

  it("reports an unreachable API with a way to start it, and retries", async () => {
    const api = await startApi(() => ({ status: 200, body: response() }));
    const url = api.url;
    await api.close(); // nothing listens any more
    const user = userEvent.setup();
    render(<DebuggerApp apiBaseUrl={url} />);
    await ask(user);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Can't reach the Neural Debugger API");
    expect(alert).toHaveTextContent("make dev-backend");
    expect(within(alert).getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("shows a busy server as retryable and a rate-limited model in plain words", async () => {
    let n = 0;
    const { user } = await setup(() => {
      n += 1;
      return n === 1
        ? { status: 429, body: { error: { code: "busy", message: "Too many answers" } } }
        : { status: 200, body: response() };
    });
    await ask(user);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The server is busy");
    await user.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("region", { name: "Likely cause" })).toBeInTheDocument();
  });

  it("can cancel a slow request", async () => {
    const { user } = await setup(() => ({ status: 200, body: response(), delayMs: 1500 }));
    await ask(user);
    await user.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(screen.queryByText(/Searching the documentation/)).toBeNull();
    expect(screen.getByRole("button", { name: "Diagnose" })).toBeEnabled();
  });

  it("keeps recent questions locally and reopens a stored answer without calling the API", async () => {
    const { api, user } = await setup(ok);
    await ask(user, "KeyError: 'email' in my script");
    await screen.findByRole("region", { name: "Likely cause" });
    const history = await screen.findByRole("region", { name: "Recent questions" });
    expect(history).toHaveTextContent("KeyError: 'email' in my script");
    expect(window.localStorage.getItem("neural-debugger:history:v1")).toContain("KeyError");

    await user.click(screen.getByRole("button", { name: "Clear" }));
    const before = api.requests.length;
    await user.click(within(history).getByRole("button", { name: /KeyError/ }));
    expect(screen.getByLabelText(/Paste a stack trace/)).toHaveValue("KeyError: 'email' in my script");
    expect(await screen.findByRole("region", { name: "Likely cause" })).toBeInTheDocument();
    expect(api.requests.length).toBe(before); // restored from this browser, no new request

    await user.click(screen.getByRole("button", { name: "Clear history" }));
    await waitFor(() => expect(screen.queryByRole("region", { name: "Recent questions" })).toBeNull());
    expect(window.localStorage.getItem("neural-debugger:history:v1")).toBeNull();
  });

  it("an abstained answer is shown as such", async () => {
    const { user } = await setup(() => ({ status: 200, body: ABSTAINED }));
    await ask(user, "TensorFlow InvalidArgumentError when calling fit()");
    expect(await screen.findByRole("status", { name: "" })).toBeTruthy();
    expect(await screen.findByText("Not enough documentation to answer")).toBeInTheDocument();
  });
});

describe("feedback", () => {
  it("one click on Helpful sends a rating without the question", async () => {
    const { api, user } = await setup(ok);
    await ask(user, "my private traceback here");
    await user.click(await screen.findByRole("button", { name: /Helpful/ }));
    expect(await screen.findByText(/your question was not stored/)).toBeInTheDocument();
    const fb = api.requests.find((r) => r.path === "/api/v1/feedback");
    expect(fb?.body).toMatchObject({ request_id: "req-123", rating: "helpful", reasons: [], include_question: false, answer_status: "answered", confidence: "medium" });
    expect((fb?.body as Record<string, unknown>).question).toBeUndefined();
  });

  it("Not helpful collects reasons and a comment, and stores the question only on explicit opt-in", async () => {
    const { api, user } = await setup(ok);
    await ask(user, "my private traceback here");
    await user.click(await screen.findByRole("button", { name: /Not helpful/ }));
    await user.click(screen.getByLabelText("The cited source does not say that"));
    await user.click(screen.getByLabelText("It is incomplete"));
    await user.type(screen.getByLabelText(/Anything else/), "E2 is about CPU tensors");
    expect(screen.getByLabelText(/Also store my question text/)).not.toBeChecked();
    await user.click(screen.getByLabelText(/Also store my question text/));
    await user.click(screen.getByRole("button", { name: "Send feedback" }));
    expect(await screen.findByText(/together with your question/)).toBeInTheDocument();
    expect(api.requests.find((r) => r.path === "/api/v1/feedback")?.body).toMatchObject({
      rating: "not_helpful",
      reasons: ["wrong_source", "incomplete"],
      comment: "E2 is about CPU tensors",
      include_question: true,
      question: "my private traceback here",
    });
  });

  it("a failed submission says so and can be retried", async () => {
    let fail = true;
    const { user } = await setup((req) =>
      req.path === "/api/v1/debug"
        ? { status: 200, body: response() }
        : fail
          ? { status: 503, body: { error: { code: "feedback_storage_full", message: "Feedback storage is full on this server." } } }
          : { status: 201, body: { id: "f", stored_question: false } },
    );
    await ask(user);
    await user.click(await screen.findByRole("button", { name: /Helpful/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Feedback storage is full");
    fail = false;
    await user.click(screen.getByRole("button", { name: /Helpful/ }));
    expect(await screen.findByText(/your feedback was saved/)).toBeInTheDocument();
  });
});
