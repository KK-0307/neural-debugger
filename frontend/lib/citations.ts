export type TextPart = { kind: "text"; text: string } | { kind: "cite"; ids: string[] };

const MARKER = /\[\s*(E\d{1,3}(?:\s*,\s*E\d{1,3})*)\s*\]/gi;

/** Split model prose into plain text and inline `[E1, E2]` citation markers. Everything stays plain text (never HTML). */
export function splitCitations(text: string): TextPart[] {
  const parts: TextPart[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const start = m.index ?? 0;
    if (start > last) parts.push({ kind: "text", text: text.slice(last, start) });
    parts.push({ kind: "cite", ids: m[1].toUpperCase().split(/\s*,\s*/) });
    last = start + m[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

/** Ids cited by a claim: the structured list first, then any inline-only ids, without duplicates. */
export function claimCitationIds(text: string, listed: string[]): string[] {
  const ids = [...listed];
  for (const p of splitCitations(text)) {
    if (p.kind === "cite") for (const id of p.ids) if (!ids.includes(id)) ids.push(id);
  }
  return ids;
}

/** Only http(s) links are ever rendered as links: documentation URLs come from the backend, but defend anyway. */
export function safeHref(url: string): string | null {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.toString() : null;
  } catch {
    return null;
  }
}

export function sourceAnchor(id: string): string {
  return `source-${id}`;
}
