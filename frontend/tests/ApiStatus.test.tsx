import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import ApiStatus from "@/components/ApiStatus";
import { DEGRADED_BODY, OK_BODY, startServer, type Reply } from "./health-server";

let close: (() => Promise<void>) | undefined;
afterEach(async () => {
  await close?.();
  close = undefined;
});

describe("ApiStatus", () => {
  it("shows a loading state, then the healthy state", async () => {
    const s = await startServer(() => ({ status: 200, body: OK_BODY }));
    close = s.close;
    render(<ApiStatus baseUrl={s.url} />);
    expect(screen.getByText("Checking API…")).toBeInTheDocument();
    expect(await screen.findByText("API healthy")).toBeInTheDocument();
    expect(screen.getByText("0.1.0")).toBeInTheDocument();
    expect(screen.getByText(/ok \(1.5 ms\)/)).toBeInTheDocument();
  });

  it("shows degraded state with the failing component", async () => {
    const s = await startServer(() => ({ status: 503, body: DEGRADED_BODY }));
    close = s.close;
    render(<ApiStatus baseUrl={s.url} />);
    expect(await screen.findByText("API degraded")).toBeInTheDocument();
    expect(screen.getByText(/OperationalError/)).toBeInTheDocument();
  });

  it("shows an error when unreachable and recovers via Retry", async () => {
    let reply: Reply = { status: 500, body: "{}" };
    const s = await startServer(() => reply);
    close = s.close;
    render(<ApiStatus baseUrl={s.url} />);
    expect(await screen.findByText("API unreachable")).toBeInTheDocument();

    reply = { status: 200, body: OK_BODY };
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("API healthy")).toBeInTheDocument();
  });
});
