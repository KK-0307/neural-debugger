"use client";

import { useCallback, useEffect, useState } from "react";
import { DEFAULT_API_BASE_URL, fetchHealth, type HealthResponse } from "@/lib/api";

type State =
  | { kind: "loading" }
  | { kind: "loaded"; health: HealthResponse }
  | { kind: "unreachable"; message: string };

export default function ApiStatus({ baseUrl = DEFAULT_API_BASE_URL }: { baseUrl?: string }) {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(baseUrl, controller.signal)
      .then((health) => setState({ kind: "loaded", health }))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setState({
          kind: "unreachable",
          message: err instanceof Error ? err.message : "Could not reach the API.",
        });
      });
    return () => controller.abort();
  }, [baseUrl, attempt]);

  const retry = useCallback(() => {
    setState({ kind: "loading" });
    setAttempt((n) => n + 1);
  }, []);

  return (
    <section
      aria-labelledby="api-status-heading"
      className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-800"
    >
      <h2 id="api-status-heading" className="text-sm font-medium text-zinc-500">
        Backend status
      </h2>
      <div role="status" aria-live="polite" className="mt-2">
        {state.kind === "loading" && <p>Checking API…</p>}
        {state.kind === "unreachable" && (
          <div className="space-y-3">
            <p className="font-medium text-red-600 dark:text-red-400">API unreachable</p>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">{state.message}</p>
            <button
              type="button"
              onClick={retry}
              className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
            >
              Retry
            </button>
          </div>
        )}
        {state.kind === "loaded" && <HealthDetails health={state.health} />}
      </div>
    </section>
  );
}

function HealthDetails({ health }: { health: HealthResponse }) {
  const ok = health.status === "ok";
  return (
    <div className="space-y-2">
      <p
        className={
          ok
            ? "font-medium text-emerald-700 dark:text-emerald-400"
            : "font-medium text-amber-700 dark:text-amber-400"
        }
      >
        {ok ? "API healthy" : "API degraded"}
      </p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-zinc-500">Version</dt>
        <dd className="font-mono">{health.version}</dd>
        <dt className="text-zinc-500">Environment</dt>
        <dd className="font-mono">{health.environment}</dd>
        {Object.entries(health.checks).map(([name, check]) => (
          <div key={name} className="contents">
            <dt className="text-zinc-500">{name}</dt>
            <dd className="font-mono">
              {check.status} ({check.latency_ms} ms)
              {check.error_type ? ` – ${check.error_type}` : ""}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
