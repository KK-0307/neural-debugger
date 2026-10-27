"use client";

import { useEffect, useRef } from "react";
import type { Source } from "@/lib/types";
import { safeHref, sourceAnchor } from "@/lib/citations";
import { Badge, SectionHeading } from "./ui";

function SourceCard({ source, active }: { source: Source; active: boolean }) {
  const ref = useRef<HTMLDetailsElement>(null);
  const href = safeHref(source.url);

  useEffect(() => {
    if (active && ref.current) ref.current.open = true;
  }, [active]);

  return (
    <article
      id={sourceAnchor(source.id)}
      className={`rounded-lg border p-3 text-sm transition-colors ${
        active
          ? "border-indigo-500 bg-indigo-50 dark:border-indigo-400 dark:bg-indigo-950/50"
          : "border-zinc-200 dark:border-zinc-800"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded border border-indigo-300 bg-indigo-50 px-1.5 font-mono text-xs font-semibold text-indigo-800 dark:border-indigo-700 dark:bg-indigo-950 dark:text-indigo-200">
          {source.id}
        </span>
        <Badge tone="info">
          {source.framework} {source.version}
        </Badge>
        {source.flags.length > 0 && (
          <Badge tone="warn" title={source.flags.join(", ")}>
            sanitised
          </Badge>
        )}
      </div>
      <p className="mt-2 font-medium">{source.title}</p>
      {source.heading && source.heading !== source.title && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400">{source.heading}</p>
      )}
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 inline-block text-indigo-700 underline underline-offset-2 hover:text-indigo-900 dark:text-indigo-300 dark:hover:text-indigo-100"
        >
          Open in the official documentation <span aria-hidden="true">↗</span>
        </a>
      ) : (
        <p className="mt-2 text-xs text-zinc-500">No safe link available.</p>
      )}
      {source.excerpt && (
        <details ref={ref} className="mt-2">
          <summary className="cursor-pointer text-xs font-medium text-zinc-600 dark:text-zinc-400">
            Text the model was shown
          </summary>
          <pre className="mt-2 max-h-64 overflow-auto rounded bg-zinc-50 p-2 text-xs leading-relaxed whitespace-pre-wrap dark:bg-zinc-900">
            {source.excerpt}
          </pre>
        </details>
      )}
    </article>
  );
}

export default function SourcesPanel({
  sources,
  activeId,
  abstained,
}: {
  sources: Source[];
  activeId: string | null;
  abstained: boolean;
}) {
  const cited = sources.filter((s) => s.cited);
  const other = sources.filter((s) => !s.cited);
  if (sources.length === 0) {
    return (
      <section aria-labelledby="sources-heading" className="space-y-2">
        <SectionHeading id="sources-heading">Sources</SectionHeading>
        <p className="text-sm text-zinc-500">No documentation was retrieved for this question.</p>
      </section>
    );
  }
  return (
    <section aria-labelledby="sources-heading" className="space-y-3">
      <SectionHeading id="sources-heading">
        {abstained ? "Closest documentation found" : "Sources"}
      </SectionHeading>
      {abstained && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          These pages were retrieved but did not contain enough to answer. They may still help you search.
        </p>
      )}
      {cited.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-zinc-500">Cited in the answer ({cited.length})</p>
          {cited.map((s) => (
            <SourceCard key={s.id} source={s} active={activeId === s.id} />
          ))}
        </div>
      )}
      {other.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-zinc-500">
            {abstained ? "Retrieved" : "Also retrieved, not cited"} ({other.length})
          </p>
          {other.map((s) => (
            <SourceCard key={s.id} source={s} active={activeId === s.id} />
          ))}
        </div>
      )}
    </section>
  );
}
