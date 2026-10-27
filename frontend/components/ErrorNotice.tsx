import { ApiError, ApiUnreachableError, DEFAULT_API_BASE_URL } from "@/lib/api";
import { safeHref } from "@/lib/citations";

type View = { title: string; body: string; hint?: string; retry: boolean };

export function describeError(err: unknown, baseUrl: string = DEFAULT_API_BASE_URL): View {
  if (err instanceof ApiUnreachableError) {
    return {
      title: "Can't reach the Neural Debugger API",
      body: err.message,
      hint: `Is the backend running? Start it with "make dev-backend" (it listens on ${baseUrl}).`,
      retry: true,
    };
  }
  if (err instanceof ApiError) {
    switch (err.code) {
      case "llm_not_configured":
        return {
          title: "No language model is configured on the server",
          body: "The documentation search worked, but there is no LLM to write the answer.",
          hint: "Set ND_LLM_PROVIDER (gemini, anthropic or openai_compatible) and that provider's API key in the server's environment, then restart it. The closest documentation is listed below.",
          retry: false,
        };
      case "busy":
        return { title: "The server is busy", body: "Too many answers are being generated right now.", hint: "Try again in a few seconds.", retry: true };
      case "llm_budget_exhausted":
        return { title: "Request budget used up", body: err.message, hint: "The server's ND_LLM_MAX_REQUESTS cap was reached.", retry: false };
      case "llm_unavailable":
        return { title: "The language model is unavailable", body: err.message, hint: "Free-tier models are rate limited and sometimes overloaded. Try again shortly.", retry: true };
      case "llm_timeout":
        return { title: "The language model took too long", body: err.message, retry: true };
      case "demo_rate_limited":
      case "demo_budget_exhausted":
        return { title: "Rate limit reached", body: err.message, hint: "This public deployment shares a small free Gemini quota; try again later or run the project locally (see the README).", retry: false };
      case "demo_question_too_long":
        return { title: "That question is too long for the public deployment", body: err.message, retry: false };
      case "validation_error":
        return { title: "That input was not accepted", body: "Check that the question is between 3 and 20,000 characters.", retry: false };
      default:
        return { title: "Something went wrong", body: err.message, hint: `HTTP ${err.status} · ${err.code}`, retry: true };
    }
  }
  return { title: "Something went wrong", body: err instanceof Error ? err.message : "Unknown error.", retry: true };
}

export default function ErrorNotice({
  error,
  baseUrl,
  onRetry,
}: {
  error: unknown;
  baseUrl?: string;
  onRetry?: () => void;
}) {
  const v = describeError(error, baseUrl);
  const sources = error instanceof ApiError ? error.sources : [];
  return (
    <div role="alert" className="space-y-3 rounded-xl border border-red-300 bg-red-50 p-4 text-red-950 dark:border-red-800 dark:bg-red-950/50 dark:text-red-100">
      <p className="font-semibold">{v.title}</p>
      <p className="text-sm">{v.body}</p>
      {v.hint && <p className="text-sm opacity-90">{v.hint}</p>}
      {sources.length > 0 && (
        <div>
          <p className="text-sm font-medium">Closest documentation found:</p>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm">
            {sources.map((s) => {
              const href = safeHref(s.url);
              return (
                <li key={s.id}>
                  {href ? (
                    <a href={href} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2">
                      {s.title}
                    </a>
                  ) : (
                    s.title
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {v.retry && onRetry && (
        <button type="button" onClick={onRetry} className="rounded-lg border border-red-400 px-3 py-1 text-sm font-medium hover:bg-red-100 dark:border-red-700 dark:hover:bg-red-900">
          Try again
        </button>
      )}
    </div>
  );
}
