export const DEFAULT_API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type ComponentStatus = {
  status: "ok" | "error";
  latency_ms: number;
  error_type?: string | null;
};

export type HealthResponse = {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  checks: Record<string, ComponentStatus>;
};

export class ApiUnreachableError extends Error {
  constructor(message = "Could not reach the API.") {
    super(message);
    this.name = "ApiUnreachableError";
  }
}

/**
 * GET /health. The backend answers 503 with a valid body when a dependency is
 * down; that is a "degraded" result, not an exception. Only network failures
 * and non-health responses throw.
 */
export async function fetchHealth(
  baseUrl: string = DEFAULT_API_BASE_URL,
  signal?: AbortSignal,
): Promise<HealthResponse> {
  let res: Response;
  try {
    res = await fetch(`${baseUrl}/health`, { signal, cache: "no-store" });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (res.status !== 200 && res.status !== 503) {
    throw new ApiUnreachableError(`Unexpected response from API (HTTP ${res.status}).`);
  }
  const body: unknown = await res.json().catch(() => null);
  if (!isHealthResponse(body)) {
    throw new ApiUnreachableError("API returned an unrecognised health payload.");
  }
  return body;
}

function isHealthResponse(v: unknown): v is HealthResponse {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    (o.status === "ok" || o.status === "degraded") &&
    typeof o.version === "string" &&
    typeof o.environment === "string" &&
    typeof o.checks === "object" &&
    o.checks !== null
  );
}
