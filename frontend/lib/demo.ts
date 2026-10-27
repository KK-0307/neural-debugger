import type { AnswerStatus, DebugResponse, Framework } from "./types";
import data from "./demo-data.json";

/** True only in the static recorded-demo build (`NEXT_PUBLIC_DEMO_MODE=static`): no backend, no key, nothing is sent anywhere. */
export const DEMO_STATIC = process.env.NEXT_PUBLIC_DEMO_MODE === "static";

export type DemoExample = {
  id: string;
  title: string;
  framework: Framework | null;
  question: string;
  /** why this example is here (the demo includes refusals and weak answers on purpose) */
  note: string;
  response: DebugResponse;
};

export type DemoMeta = {
  model: string;
  recorded_on: string;
  pipeline_commit: string;
  note: string;
};

export const DEMO_EXAMPLES = (data as unknown as { examples: DemoExample[] }).examples;
export const DEMO_META = (data as unknown as { meta: Partial<DemoMeta> }).meta;

export function outcomeLabel(status: AnswerStatus): string {
  return { answered: "Answered", partial: "Partly answered", abstained: "Declined to answer", error: "Error" }[status];
}
