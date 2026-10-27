"use client";

import type { Claim } from "@/lib/types";
import { claimCitationIds, splitCitations } from "@/lib/citations";
import { claimMark } from "@/lib/uncertainty";
import { Badge } from "./ui";

export function CitationChip({ id, onCite }: { id: string; onCite: (id: string) => void }) {
  return (
    <button
      type="button"
      onClick={() => onCite(id)}
      aria-label={`Show source ${id}`}
      className="mx-0.5 inline-flex items-center rounded border border-indigo-300 bg-indigo-50 px-1.5 py-0 align-baseline font-mono text-xs font-semibold text-indigo-800 hover:bg-indigo-100 focus-visible:outline-2 focus-visible:outline-indigo-500 dark:border-indigo-700 dark:bg-indigo-950 dark:text-indigo-200 dark:hover:bg-indigo-900"
    >
      {id}
    </button>
  );
}

/** One statement of the answer: plain text (never HTML), inline citation chips, and a check badge if it needs attention. */
export default function ClaimText({ claim, onCite }: { claim: Claim; onCite: (id: string) => void }) {
  const inline = new Set(splitCitations(claim.text).flatMap((p) => (p.kind === "cite" ? p.ids : [])));
  const trailing = claimCitationIds(claim.text, claim.citations).filter((id) => !inline.has(id));
  const mark = claimMark(claim);
  return (
    <span>
      {splitCitations(claim.text).map((p, i) =>
        p.kind === "text" ? (
          <span key={i}>{p.text}</span>
        ) : (
          p.ids.map((id) => <CitationChip key={`${i}-${id}`} id={id} onCite={onCite} />)
        ),
      )}
      {trailing.map((id) => (
        <CitationChip key={`t-${id}`} id={id} onCite={onCite} />
      ))}
      {mark && mark.tone !== "good" && (
        <span className="ml-1.5 align-baseline">
          <Badge tone={mark.tone} title={mark.hint}>
            {mark.label}
          </Badge>
        </span>
      )}
    </span>
  );
}
