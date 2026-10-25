import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";

export type Reply = { status: number; body: string };

/** A real local HTTP server, so tests exercise the real fetch + parsing path. */
export async function startServer(
  reply: () => Reply,
): Promise<{ url: string; close: () => Promise<void>; server: Server }> {
  const server = createServer((_req, res) => {
    const { status, body } = reply();
    res.writeHead(status, { "content-type": "application/json" });
    res.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    server,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

export const OK_BODY = JSON.stringify({
  status: "ok",
  version: "0.1.0",
  environment: "test",
  checks: { database: { status: "ok", latency_ms: 1.5 } },
});

export const DEGRADED_BODY = JSON.stringify({
  status: "degraded",
  version: "0.1.0",
  environment: "test",
  checks: { database: { status: "error", latency_ms: 2.0, error_type: "OperationalError" } },
});
