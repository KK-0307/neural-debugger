import type { DebugResponse, FeedbackInput, Framework } from "./types";

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

// ------------------------------------------------------------------------------------- debugging API

export const MAX_QUESTION_CHARS = 20000;
export const MIN_QUESTION_CHARS = 3;

/** A non-2xx answer from the API, with the backend's structured error code (`llm_not_configured`, `busy`, ...). */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;
  readonly sources: { id: string; title: string; url: string }[];
  constructor(
    status: number,
    code: string,
    message: string,
    requestId: string | null = null,
    sources: { id: string; title: string; url: string }[] = [],
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.sources = sources;
  }
}

async function postJson(
  path: string,
  body: unknown,
  baseUrl: string,
  signal?: AbortSignal,
): Promise<Response> {
  try {
    return await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
      cache: "no-store",
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError(`Could not reach the API at ${baseUrl}.`);
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  const body: unknown = await res.json().catch(() => null);
  const err = (body as { error?: Record<string, unknown> } | null)?.error;
  const details = Array.isArray(err?.details) ? (err?.details as Record<string, unknown>[]) : [];
  const sources = (details[0]?.sources ?? []) as { id: string; title: string; url: string }[];
  return new ApiError(
    res.status,
    typeof err?.code === "string" ? err.code : `http_${res.status}`,
    typeof err?.message === "string" ? err.message : `The API answered HTTP ${res.status}.`,
    typeof err?.request_id === "string" ? err.request_id : null,
    Array.isArray(sources) ? sources : [],
  );
}

export async function askDebugger(
  question: string,
  framework: Framework | null,
  baseUrl: string = DEFAULT_API_BASE_URL,
  signal?: AbortSignal,
): Promise<DebugResponse> {
  const res = await postJson("/api/v1/debug", { question, framework }, baseUrl, signal);
  if (!res.ok) throw await toApiError(res);
  const body: unknown = await res.json().catch(() => null);
  if (!isDebugResponse(body)) {
    throw new ApiError(res.status, "bad_response", "The API returned an unrecognised answer.");
  }
  return body;
}

export async function sendFeedback(
  input: FeedbackInput,
  baseUrl: string = DEFAULT_API_BASE_URL,
  signal?: AbortSignal,
): Promise<void> {
  const res = await postJson("/api/v1/feedback", input, baseUrl, signal);
  if (!res.ok) throw await toApiError(res);
}

export function isDebugResponse(v: unknown): v is DebugResponse {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.status === "string" &&
    typeof o.confidence === "string" &&
    Array.isArray(o.steps) &&
    Array.isArray(o.sources) &&
    Array.isArray(o.citations) &&
    typeof o.validation === "object" &&
    o.validation !== null
  );
}
