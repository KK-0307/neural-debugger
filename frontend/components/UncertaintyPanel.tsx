import type { DebugResponse } from "@/lib/types";
import { explainConfidence, provenanceNote } from "@/lib/uncertainty";
import { Badge, SectionHeading, TONE_DOT, toneClasses } from "./ui";

const STATUS_TEXT: Record<DebugResponse["status"], string> = {
  answered: "Answered",
  partial: "Partly answered",
  abstained: "Not enough documentation to answer",
  error: "Error",
};

export function StatusBanner({ result }: { result: DebugResponse }) {
  const e = explainConfidence(result);
  return (
    <div
      role="status"
      className={`flex flex-wrap items-center justify-between gap-3 rounded-xl border px-4 py-3 ${toneClasses(e.tone)}`}
    >
      <div>
        <p className="font-semibold">{STATUS_TEXT[result.status]}</p>
        <p className="text-sm opacity-90">{e.summary}</p>
      </div>
      <Badge tone={e.tone}>{e.headline}</Badge>
    </div>
  );
}

export default function UncertaintyPanel({ result }: { result: DebugResponse }) {
  const e = explainConfidence(result);
  const gate = result.evidence_gate;
  return (
    <section aria-labelledby="uncertainty-heading" className="space-y-3">
      <SectionHeading id="uncertainty-heading">How sure is this?</SectionHeading>
      <details open={result.confidence !== "high"} className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
        <summary className="cursor-pointer text-sm font-medium">
          {e.headline}
          {gate ? ` · documentation coverage ${Math.round(gate.coverage * 100)}%` : ""}
        </summary>
        <ul className="mt-3 space-y-2 text-sm">
          {e.reasons.map((r, i) => (
            <li key={i} className="flex gap-2">
              <span aria-hidden="true" className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${TONE_DOT[r.tone]}`} />
              <span>{r.text}</span>
            </li>
          ))}
          {e.reasons.length === 0 && <li className="text-zinc-500">No validation warnings were raised.</li>}
        </ul>
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 border-t border-zinc-200 pt-3 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
          <dt>Statements checked</dt>
          <dd>{result.validation.total_claims}</dd>
          <dt>Model</dt>
          <dd className="font-mono">{result.model ?? result.provider ?? "none"}</dd>
          <dt>Time</dt>
          <dd>
            {(result.timings.total_ms / 1000).toFixed(1)} s (search {Math.round(result.timings.retrieval_ms)} ms,
            model {(result.timings.generation_ms / 1000).toFixed(1)} s)
          </dd>
        </dl>
      </details>
      <p className="text-xs text-zinc-500 dark:text-zinc-400">
        {provenanceNote(result)}
      </p>
    </section>
  );
}
