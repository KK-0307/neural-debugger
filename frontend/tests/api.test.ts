import { afterEach, describe, expect, it } from "vitest";
import { ApiUnreachableError, fetchHealth } from "@/lib/api";
import { DEGRADED_BODY, OK_BODY, startServer } from "./health-server";

let close: (() => Promise<void>) | undefined;
afterEach(async () => {
  await close?.();
  close = undefined;
});

describe("fetchHealth", () => {
  it("parses a healthy response", async () => {
    const s = await startServer(() => ({ status: 200, body: OK_BODY }));
    close = s.close;
    const h = await fetchHealth(s.url);
    expect(h.status).toBe("ok");
    expect(h.checks.database.status).toBe("ok");
  });

  it("treats a 503 with a valid body as degraded, not an error", async () => {
    const s = await startServer(() => ({ status: 503, body: DEGRADED_BODY }));
    close = s.close;
    const h = await fetchHealth(s.url);
    expect(h.status).toBe("degraded");
    expect(h.checks.database.error_type).toBe("OperationalError");
  });

  it("throws ApiUnreachableError on unexpected status", async () => {
    const s = await startServer(() => ({ status: 500, body: "{}" }));
    close = s.close;
    await expect(fetchHealth(s.url)).rejects.toThrow(ApiUnreachableError);
  });

  it("throws ApiUnreachableError on a malformed payload", async () => {
    const s = await startServer(() => ({ status: 200, body: '{"hello":"world"}' }));
    close = s.close;
    await expect(fetchHealth(s.url)).rejects.toThrow(/unrecognised/);
  });

  it("throws ApiUnreachableError when nothing is listening", async () => {
    const s = await startServer(() => ({ status: 200, body: OK_BODY }));
    await s.close(); // port is now closed
    await expect(fetchHealth(s.url)).rejects.toThrow(ApiUnreachableError);
  });
});
