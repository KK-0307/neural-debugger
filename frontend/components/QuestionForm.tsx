"use client";

import { useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { MAX_QUESTION_CHARS, MIN_QUESTION_CHARS } from "@/lib/api";
import type { Framework } from "@/lib/types";
import { buttonPrimary, buttonSecondary } from "./ui";

export const EXAMPLES: { label: string; framework: Framework | null; text: string }[] = [
  {
    label: "PyTorch: tensors on different devices",
    framework: "pytorch",
    text: "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!",
  },
  {
    label: "FastAPI: 422 on a JSON body",
    framework: "fastapi",
    text: "I POST JSON to my FastAPI endpoint and get 422 Unprocessable Entity with 'Field required' for body. What is wrong?",
  },
  {
    label: "Python: coroutine never awaited",
    framework: "python",
    text: "RuntimeWarning: coroutine 'main' was never awaited. Why does this appear and how do I fix it?",
  },
];

export type Submission = { question: string; framework: Framework | null };

export default function QuestionForm({
  onSubmit,
  onCancel,
  busy,
  initial,
}: {
  onSubmit: (s: Submission) => void;
  onCancel: () => void;
  busy: boolean;
  initial?: Submission;
}) {
  const uid = useId();
  const [question, setQuestion] = useState(initial?.question ?? "");
  const [framework, setFramework] = useState<Framework | "">(initial?.framework ?? "");
  const ref = useRef<HTMLTextAreaElement>(null);

  const length = question.length;
  const tooShort = question.trim().length < MIN_QUESTION_CHARS;
  const tooLong = length > MAX_QUESTION_CHARS;
  const canSubmit = !busy && !tooShort && !tooLong;

  function submit(e?: FormEvent) {
    e?.preventDefault();
    if (canSubmit) onSubmit({ question: question.trim(), framework: framework || null });
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
  }

  return (
    <form onSubmit={submit} className="space-y-3" aria-describedby={`${uid}-privacy`}>
      <div>
        <label htmlFor={`${uid}-q`} className="text-sm font-semibold">
          Paste a stack trace, an error message, or describe the problem
        </label>
        <textarea
          id={`${uid}-q`}
          ref={ref}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          rows={9}
          spellCheck={false}
          placeholder={'Traceback (most recent call last):\n  File "train.py", line 52, in <module>\n…\nRuntimeError: …'}
          aria-invalid={tooLong}
          className="mt-1 block w-full resize-y rounded-xl border border-zinc-300 bg-white p-3 font-mono text-sm leading-relaxed shadow-sm placeholder:text-zinc-400 focus-visible:outline-2 focus-visible:outline-indigo-500 dark:border-zinc-700 dark:bg-zinc-950"
        />
        <div className="mt-1 flex justify-between text-xs text-zinc-500">
          <span>
            <kbd className="rounded border px-1">Ctrl</kbd>/<kbd className="rounded border px-1">⌘</kbd> +{" "}
            <kbd className="rounded border px-1">Enter</kbd> to submit
          </span>
          <span className={tooLong ? "text-red-600 dark:text-red-400" : ""}>
            {length.toLocaleString()} / {MAX_QUESTION_CHARS.toLocaleString()}
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label htmlFor={`${uid}-fw`} className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
            Documentation
          </label>
          <select
            id={`${uid}-fw`}
            value={framework}
            onChange={(e) => setFramework(e.target.value as Framework | "")}
            className="mt-1 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
          >
            <option value="">Auto (search all)</option>
            <option value="python">Python 3.12</option>
            <option value="pytorch">PyTorch 2.14</option>
            <option value="fastapi">FastAPI 0.162</option>
          </select>
        </div>
        <button type="submit" className={buttonPrimary} disabled={!canSubmit}>
          {busy ? "Working…" : "Diagnose"}
        </button>
        {busy && (
          <button type="button" className={buttonSecondary} onClick={onCancel}>
            Cancel
          </button>
        )}
        {!busy && question && (
          <button type="button" className={buttonSecondary} onClick={() => { setQuestion(""); ref.current?.focus(); }}>
            Clear
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-zinc-500">Try an example:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            disabled={busy}
            onClick={() => {
              setQuestion(ex.text);
              setFramework(ex.framework ?? "");
              ref.current?.focus();
            }}
            className="rounded-full border border-zinc-300 px-3 py-1 hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
          >
            {ex.label}
          </button>
        ))}
      </div>

      <p id={`${uid}-privacy`} className="text-xs text-zinc-500 dark:text-zinc-400">
        Your text is sent to the Neural Debugger server and on to its configured language model provider. Do not paste
        passwords, API keys or private code you cannot share.
      </p>
    </form>
  );
}
