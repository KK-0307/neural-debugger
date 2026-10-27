"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULT_API_BASE_URL, askDebugger, fetchPublicConfig, type PublicConfig } from "@/lib/api";
import { addEntry, clearHistory, loadHistory, saveHistory, useHistory, type HistoryEntry } from "@/lib/history";
import type { DebugResponse } from "@/lib/types";
import AnswerView from "./AnswerView";
import ErrorNotice from "./ErrorNotice";
import HistoryPanel from "./HistoryPanel";
import QuestionForm, { type Submission } from "./QuestionForm";

type State =
  | { kind: "idle" }
  | { kind: "loading"; since: number }
  | { kind: "result"; question: string; result: DebugResponse; key: string }
  | { kind: "error"; error: unknown; last: Submission };

export default function DebuggerApp({ apiBaseUrl = DEFAULT_API_BASE_URL }: { apiBaseUrl?: string }) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const history = useHistory();
  const [elapsed, setElapsed] = useState(0);
  const [formKey, setFormKey] = useState(0);
  const [formInitial, setFormInitial] = useState<Submission | undefined>(undefined);
  const controller = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const [config, setConfig] = useState<PublicConfig | null>(null);

  useEffect(() => {
    const c = new AbortController();
    fetchPublicConfig(apiBaseUrl, c.signal).then((cfg) => {
      if (!c.signal.aborted) setConfig(cfg);
    });
    return () => c.abort();
  }, [apiBaseUrl]);

  useEffect(() => {
    if (state.kind !== "loading") return;
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - state.since) / 1000)), 500);
    return () => clearInterval(t);
  }, [state]);

  useEffect(() => {
    if (state.kind === "result" || state.kind === "error") resultRef.current?.focus();
  }, [state]);

  const submit = useCallback(
    async (s: Submission) => {
      controller.current?.abort();
      const c = new AbortController();
      controller.current = c;
      setElapsed(0);
      setState({ kind: "loading", since: Date.now() });
      try {
        const result = await askDebugger(s.question, s.framework, apiBaseUrl, c.signal);
        const key = result.request_id ?? String(Date.now());
        setState({ kind: "result", question: s.question, result, key });
        saveHistory(addEntry(loadHistory(), { id: key, at: Date.now(), question: s.question, framework: s.framework, result }));
      } catch (err) {
        if (c.signal.aborted) return;
        setState({ kind: "error", error: err, last: s });
      }
    },
    [apiBaseUrl],
  );

  const cancel = useCallback(() => {
    controller.current?.abort();
    setState({ kind: "idle" });
  }, []);

  const openEntry = useCallback((e: HistoryEntry) => {
    controller.current?.abort();
    setFormInitial({ question: e.question, framework: e.framework });
    setFormKey((k) => k + 1);
    setState({ kind: "result", question: e.question, result: e.result, key: e.id });
  }, []);

  const busy = state.kind === "loading";
  return (
    <div className="space-y-8">
      <QuestionForm key={formKey} initial={formInitial} onSubmit={submit} onCancel={cancel} busy={busy} />
      {config && <DeploymentNotice config={config} />}

      <div ref={resultRef} tabIndex={-1} className="space-y-4 outline-none" aria-live="polite" aria-busy={busy}>
        {state.kind === "loading" && (
          <div role="status" className="flex items-center gap-3 rounded-xl border border-zinc-200 p-4 text-sm dark:border-zinc-800">
            <span aria-hidden="true" className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
            <span>
              Searching the documentation and writing an answer… {elapsed}s
              <span className="block text-xs text-zinc-500">Free-tier models can take 10–30 seconds.</span>
            </span>
          </div>
        )}
        {state.kind === "error" && <ErrorNotice error={state.error} baseUrl={apiBaseUrl} onRetry={() => submit(state.last)} />}
        {state.kind === "result" && (
          <AnswerView
            key={state.key}
            result={state.result}
            question={state.question}
            apiBaseUrl={apiBaseUrl}
            readOnly={config?.mode === "readonly"}
          />
        )}
      </div>

      <HistoryPanel
        entries={history}
        onOpen={openEntry}
        onClear={clearHistory}
      />
    </div>
  );
}

function DeploymentNotice({ config }: { config: PublicConfig }) {
  if (config.mode !== "readonly" && !config.notice) return null;
  return (
    <aside
      aria-label="About this public deployment"
      className="space-y-1 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-100"
    >
      {config.mode === "readonly" && (
        <p>
          <strong>Public deployment on free hosting.</strong> It is shared and rate limited
          {config.rate_per_minute !== null && config.rate_per_day !== null
            ? ` (${config.rate_per_minute} questions per minute and ${config.rate_per_day} per day per visitor)`
            : ""}
          {config.max_question_chars !== null ? `; questions up to ${config.max_question_chars.toLocaleString("en-US")} characters` : ""}
          . Feedback is turned off and nothing you type is stored by this server. The first request after a quiet period can take a
          minute while the server wakes up.
        </p>
      )}
      {config.notice && <p>{config.notice}</p>}
    </aside>
  );
}
