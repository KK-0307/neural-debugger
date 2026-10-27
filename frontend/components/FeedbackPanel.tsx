"use client";

import { useId, useState } from "react";
import { DEFAULT_API_BASE_URL, sendFeedback } from "@/lib/api";
import type { DebugResponse, FeedbackReason } from "@/lib/types";
import { SectionHeading, buttonPrimary, buttonSecondary } from "./ui";

const REASONS: { id: FeedbackReason; label: string }[] = [
  { id: "wrong_answer", label: "The answer is wrong" },
  { id: "incomplete", label: "It is incomplete" },
  { id: "wrong_source", label: "The cited source does not say that" },
  { id: "unclear", label: "It is unclear" },
  { id: "should_have_answered", label: "It should have answered" },
  { id: "should_have_abstained", label: "It should not have answered" },
  { id: "other", label: "Something else" },
];

type Phase = "idle" | "details" | "sending" | "sent" | "failed";

export default function FeedbackPanel({
  result,
  question,
  apiBaseUrl = DEFAULT_API_BASE_URL,
}: {
  result: DebugResponse;
  question: string;
  apiBaseUrl?: string;
}) {
  const uid = useId();
  const [phase, setPhase] = useState<Phase>("idle");
  const [rating, setRating] = useState<"helpful" | "not_helpful" | null>(null);
  const [reasons, setReasons] = useState<FeedbackReason[]>([]);
  const [comment, setComment] = useState("");
  const [includeQuestion, setIncludeQuestion] = useState(false);
  const [error, setError] = useState("");

  if (!result.request_id) return null; // nothing to attach feedback to

  async function submit(r: "helpful" | "not_helpful") {
    setRating(r);
    setPhase("sending");
    setError("");
    try {
      await sendFeedback(
        {
          request_id: result.request_id as string,
          rating: r,
          reasons: r === "not_helpful" ? reasons : [],
          comment,
          answer_status: result.status,
          confidence: result.confidence,
          include_question: includeQuestion,
          question: includeQuestion ? question : undefined,
        },
        apiBaseUrl,
      );
      setPhase("sent");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send feedback.");
      setPhase("failed");
    }
  }

  function toggle(id: FeedbackReason) {
    setReasons((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));
  }

  return (
    <section aria-labelledby={`${uid}-h`} className="space-y-3 rounded-xl border border-dashed border-zinc-300 p-4 dark:border-zinc-700">
      <SectionHeading id={`${uid}-h`}>Was this useful?</SectionHeading>

      {phase === "sent" ? (
        <p role="status" className="text-sm text-emerald-700 dark:text-emerald-300">
          Thanks, your feedback was saved{includeQuestion ? " together with your question" : " (your question was not stored)"}.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className={buttonSecondary}
              aria-pressed={rating === "helpful"}
              disabled={phase === "sending"}
              onClick={() => submit("helpful")}
            >
              <span aria-hidden="true">👍</span> Helpful
            </button>
            <button
              type="button"
              className={buttonSecondary}
              aria-pressed={phase === "details" || rating === "not_helpful"}
              disabled={phase === "sending"}
              onClick={() => {
                setRating("not_helpful");
                setPhase("details");
              }}
            >
              <span aria-hidden="true">👎</span> Not helpful
            </button>
          </div>

          {(phase === "details" || phase === "failed" || phase === "sending") && rating === "not_helpful" && (
            <div className="space-y-3">
              <fieldset className="space-y-1">
                <legend className="text-sm font-medium">What went wrong?</legend>
                {REASONS.map((r) => (
                  <label key={r.id} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={reasons.includes(r.id)} onChange={() => toggle(r.id)} />
                    {r.label}
                  </label>
                ))}
              </fieldset>
              <div>
                <label htmlFor={`${uid}-c`} className="text-sm font-medium">
                  Anything else? <span className="font-normal text-zinc-500">(optional, {1000 - comment.length} characters left)</span>
                </label>
                <textarea
                  id={`${uid}-c`}
                  value={comment}
                  maxLength={1000}
                  onChange={(e) => setComment(e.target.value)}
                  rows={3}
                  className="mt-1 w-full rounded-lg border border-zinc-300 bg-white p-2 text-sm dark:border-zinc-700 dark:bg-zinc-900"
                />
              </div>
              <label className="flex items-start gap-2 text-sm">
                <input
                  type="checkbox"
                  className="mt-1"
                  checked={includeQuestion}
                  onChange={(e) => setIncludeQuestion(e.target.checked)}
                />
                <span>
                  Also store my question text so the problem can be reproduced.
                  <span className="block text-xs text-zinc-500">
                    Off by default. Leave it off if your question contains private code or secrets.
                  </span>
                </span>
              </label>
              <button type="button" className={buttonPrimary} disabled={phase === "sending"} onClick={() => submit("not_helpful")}>
                {phase === "sending" ? "Sending…" : "Send feedback"}
              </button>
            </div>
          )}
          {phase === "failed" && (
            <p role="alert" className="text-sm text-red-700 dark:text-red-300">
              {error}
            </p>
          )}
        </>
      )}
    </section>
  );
}
