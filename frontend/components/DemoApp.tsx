"use client";

import { useState } from "react";
import { DEMO_EXAMPLES, DEMO_META, outcomeLabel, type DemoExample } from "@/lib/demo";
import AnswerView from "./AnswerView";
import { Badge, SectionHeading, card } from "./ui";

const TONE = { answered: "good", partial: "info", abstained: "bad", error: "bad" } as const;

export default function DemoApp({ examples = DEMO_EXAMPLES }: { examples?: DemoExample[] }) {
  const [selected, setSelected] = useState<DemoExample | null>(null);

  return (
    <div className="space-y-8">
      <aside
        role="note"
        className="space-y-1 rounded-xl border border-indigo-300 bg-indigo-50 p-4 text-sm text-indigo-950 dark:border-indigo-800 dark:bg-indigo-950/50 dark:text-indigo-100"
      >
        <p className="font-semibold">Recorded demo: read-only, no server behind it</p>
        <p>
          Every answer below is a real output of {DEMO_META.model ?? "a hosted language model"} recorded on{" "}
          {DEMO_META.recorded_on ?? "an earlier date"}, replayed through the project&apos;s validators. Nothing you click is sent
          anywhere, no API key exists in this page, and feedback is disabled. The examples were chosen by me to show
          behaviour, <strong>including a refusal and a weak answer</strong>; they are not a sample. To ask your own question, run the
          project locally (see the README).
        </p>
      </aside>

      <section aria-labelledby="demo-examples" className="space-y-3">
        <SectionHeading id="demo-examples">Pick a recorded question</SectionHeading>
        {examples.length === 0 && <p className="text-sm text-zinc-500">No recorded examples are bundled in this build.</p>}
        <ul className="grid gap-3 sm:grid-cols-2">
          {examples.map((ex) => (
            <li key={ex.id}>
              <button
                type="button"
                onClick={() => setSelected(ex)}
                aria-pressed={selected?.id === ex.id}
                className={`${card} h-full w-full space-y-2 text-left transition-colors hover:border-indigo-400 focus-visible:outline-2 focus-visible:outline-indigo-500 ${
                  selected?.id === ex.id ? "border-indigo-500 dark:border-indigo-400" : ""
                }`}
              >
                <span className="flex flex-wrap items-center gap-2">
                  <Badge tone={TONE[ex.response.status]}>{outcomeLabel(ex.response.status)}</Badge>
                  {ex.framework && <Badge tone="info">{ex.framework}</Badge>}
                </span>
                <span className="block font-medium">{ex.title}</span>
                <span className="block text-xs text-zinc-500 dark:text-zinc-400">{ex.note}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      {selected && (
        <div className="space-y-4" aria-live="polite">
          <details className="rounded-xl border border-zinc-200 p-3 dark:border-zinc-800" open>
            <summary className="cursor-pointer text-sm font-semibold">The question that was asked</summary>
            <pre className="mt-2 max-h-72 overflow-auto font-mono text-sm leading-relaxed whitespace-pre-wrap">{selected.question}</pre>
          </details>
          <AnswerView key={selected.id} result={selected.response} question={selected.question} readOnly />
        </div>
      )}
    </div>
  );
}
