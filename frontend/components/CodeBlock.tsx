"use client";

import { useState } from "react";
import { buttonSecondary } from "./ui";

export default function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState<"idle" | "ok" | "failed">("idle");

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied("ok");
    } catch {
      setCopied("failed");
    }
    setTimeout(() => setCopied("idle"), 2000);
  }

  return (
    <figure className="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800">
      <figcaption className="flex items-center justify-between gap-3 border-b border-zinc-200 bg-zinc-50 px-3 py-1.5 text-xs text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
        <span>
          <span className="font-mono font-semibold">{language || "text"}</span>
          <span className="ml-2">illustrative · not run by Neural Debugger</span>
        </span>
        <button
          type="button"
          onClick={copy}
          className={`${buttonSecondary} !px-2 !py-0.5 !text-xs`}
          aria-label="Copy code to clipboard"
        >
          {copied === "ok" ? "Copied" : copied === "failed" ? "Copy failed" : "Copy"}
        </button>
      </figcaption>
      <pre className="overflow-x-auto bg-white p-3 text-sm leading-relaxed dark:bg-zinc-950">
        <code className="font-mono">{code}</code>
      </pre>
    </figure>
  );
}
