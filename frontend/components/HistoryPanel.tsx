"use client";

import type { HistoryEntry } from "@/lib/history";
import { SectionHeading } from "./ui";

const LABEL = { answered: "answered", partial: "partial", abstained: "no answer", error: "error" } as const;

export default function HistoryPanel({
  entries,
  onOpen,
  onClear,
}: {
  entries: HistoryEntry[];
  onOpen: (e: HistoryEntry) => void;
  onClear: () => void;
}) {
  if (entries.length === 0) return null;
  return (
    <section aria-labelledby="history-heading" className="space-y-2">
      <div className="flex items-center justify-between">
        <SectionHeading id="history-heading">Recent questions</SectionHeading>
        <button type="button" onClick={onClear} className="text-xs text-zinc-500 underline underline-offset-2 hover:text-zinc-800 dark:hover:text-zinc-200">
          Clear history
        </button>
      </div>
      <ul className="space-y-1">
        {entries.map((e) => (
          <li key={e.id}>
            <button
              type="button"
              onClick={() => onOpen(e)}
              className="flex w-full items-baseline justify-between gap-3 rounded-lg border border-transparent px-2 py-1.5 text-left text-sm hover:border-zinc-200 hover:bg-zinc-50 dark:hover:border-zinc-800 dark:hover:bg-zinc-900"
            >
              <span className="truncate font-mono text-xs">{e.question.split("\n").find((l) => l.trim()) ?? e.question}</span>
              <span className="shrink-0 text-xs text-zinc-500">{LABEL[e.result.status]}</span>
            </button>
          </li>
        ))}
      </ul>
      <p className="text-xs text-zinc-500">Kept only in this browser (localStorage). Nothing is uploaded.</p>
    </section>
  );
}
