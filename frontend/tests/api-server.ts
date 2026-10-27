import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";

export type Recorded = { method: string; path: string; body: unknown };
export type Reply = { status: number; body: unknown; delayMs?: number };
export type Router = (req: Recorded) => Reply;

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const text = Buffer.concat(chunks).toString("utf8");
  try {
    return text ? JSON.parse(text) : undefined;
  } catch {
    return text;
  }
}

/** A real local HTTP server with a per-test router; records every request so tests can assert on payloads. */
export async function startApi(router: Router) {
  const requests: Recorded[] = [];
  const server = createServer(async (req, res) => {
    const rec: Recorded = { method: req.method ?? "GET", path: req.url ?? "/", body: await readJson(req) };
    requests.push(rec);
    const reply = router(rec);
    if (reply.delayMs) await new Promise((r) => setTimeout(r, reply.delayMs));
    res.writeHead(reply.status, { "content-type": "application/json" });
    res.end(typeof reply.body === "string" ? reply.body : JSON.stringify(reply.body));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    close: () => new Promise<void>((resolve) => { server.closeAllConnections(); server.close(() => resolve()); }),
  };
}
