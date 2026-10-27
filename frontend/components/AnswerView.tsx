"use client";

import { useCallback, useState } from "react";
import type { DebugResponse } from "@/lib/types";
import { sourceAnchor } from "@/lib/citations";
import { visibleCaveats } from "@/lib/uncertainty";
import ClaimText from "./ClaimText";
import CodeBlock from "./CodeBlock";
import FeedbackPanel from "./FeedbackPanel";
import SourcesPanel from "./SourcesPanel";
import UncertaintyPanel, { StatusBanner } from "./UncertaintyPanel";
import { SectionHeading, card } from "./ui";

const CAVEATS_SHOWN = 3;

export default function AnswerView({
  result,
  question,
  apiBaseUrl,
}: {
  result: DebugResponse;
  question: string;
  apiBaseUrl?: string;
}) {
  const [activeSource, setActiveSource] = useState<string | null>(null);
  const [showAllCaveats, setShowAllCaveats] = useState(false);

  const onCite = useCallback((id: string) => {
    setActiveSource(id);
    document.getElementById(sourceAnchor(id))?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  }, []);

  const abstained = result.status === "abstained";
  const allCaveats = visibleCaveats(result);
  const caveats = showAllCaveats ? allCaveats : allCaveats.slice(0, CAVEATS_SHOWN);

  return (
    <div className="space-y-5">
      <StatusBanner result={result} />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="space-y-6">
          {result.diagnosis && (
            <section aria-labelledby="diagnosis-heading" className={`${card} space-y-2`}>
              <SectionHeading id="diagnosis-heading">Likely cause</SectionHeading>
              <p className="text-base leading-relaxed">
                <ClaimText claim={result.diagnosis} onCite={onCite} />
              </p>
            </section>
          )}

          {result.steps.length > 0 && (
            <section aria-labelledby="steps-heading" className={`${card} space-y-3`}>
              <SectionHeading id="steps-heading">Suggested fixes</SectionHeading>
              <ol className="list-decimal space-y-3 pl-5 marker:font-semibold marker:text-indigo-600 dark:marker:text-indigo-300">
                {result.steps.map((s, i) => (
                  <li key={i} className="pl-1 leading-relaxed">
                    <ClaimText claim={s} onCite={onCite} />
                  </li>
                ))}
              </ol>
            </section>
          )}

          {result.code_suggestions.length > 0 && (
            <section aria-labelledby="code-heading" className="space-y-3">
              <SectionHeading id="code-heading">Example code</SectionHeading>
              {result.code_suggestions.map((c, i) => (
                <div key={i} className="space-y-2">
                  <p className="text-sm leading-relaxed">
                    <ClaimText claim={c.explanation} onCite={onCite} />
                  </p>
                  <CodeBlock language={c.language} code={c.code} />
                </div>
              ))}
            </section>
          )}

          {result.conflicts.length > 0 && (
            <section aria-labelledby="conflicts-heading" className={`${card} space-y-2`}>
              <SectionHeading id="conflicts-heading">The documentation disagrees</SectionHeading>
              <ul className="list-disc space-y-2 pl-5">
                {result.conflicts.map((c, i) => (
                  <li key={i}>
                    <ClaimText claim={c} onCite={onCite} />
                  </li>
                ))}
              </ul>
            </section>
          )}

          {result.missing_information.length > 0 && (
            <section aria-labelledby="missing-heading" className={`${card} space-y-2`}>
              <SectionHeading id="missing-heading">What the documentation did not cover</SectionHeading>
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {result.missing_information.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            </section>
          )}

          {allCaveats.length > 0 && (
            <section aria-labelledby="caveats-heading" className="space-y-2">
              <SectionHeading id="caveats-heading">Keep in mind</SectionHeading>
              <ul className="list-disc space-y-1 pl-5 text-sm text-zinc-700 dark:text-zinc-300">
                {caveats.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
              {allCaveats.length > CAVEATS_SHOWN && (
                <button
                  type="button"
                  onClick={() => setShowAllCaveats((v) => !v)}
                  className="text-sm text-indigo-700 underline underline-offset-2 dark:text-indigo-300"
                >
                  {showAllCaveats ? "Show fewer" : `Show ${allCaveats.length - CAVEATS_SHOWN} more`}
                </button>
              )}
            </section>
          )}

          <UncertaintyPanel result={result} />
          <FeedbackPanel result={result} question={question} apiBaseUrl={apiBaseUrl} />
        </div>

        <aside aria-label="Source documents" className="lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:self-start lg:overflow-y-auto">
          <SourcesPanel sources={result.sources} activeId={activeSource} abstained={abstained} />
        </aside>
      </div>
    </div>
  );
}
