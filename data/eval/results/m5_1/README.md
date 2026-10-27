# Milestone 5.1 evidence (2026-10-26)

Produced by the commands in `docs/rag.md` section 11 and `docs/semantic-grounding.md`. Integrity is tracked by
`data/eval/m5_1_freeze.json` (`make freeze-verify`). **Nothing here is human-labelled.** Question sets are
*agent-authored* (`live_acceptance.yaml`) or *model-generated* (`independent_gemini.yaml`, written by Gemini); judgements of
answer quality are *agent reviews*; the adversarial set is *synthetic*.

| file | what it is |
|---|---|
| `live_gemini_gemini-3.1-flash-lite.json`, `.raw.json` | run 1 on the 21 agent-authored questions, **as recorded** (validators as they were at run time). `.raw.json` holds the prompts, the model's raw replies, errors, and the answers the pipeline produced; the API key is redacted (and was not present) |
| `rerun_infra_errors/` | the 4 questions of run 1 that ended in Google 503 errors, run again once. A rerun only fills an infrastructure error; it never replaces a model answer |
| `live_merged_as_run.json` | run 1 + rerun merged (21 of 21 served), as recorded |
| `live_rescored_{off,rules,nli}.json` | the **same recorded replies** replayed through the *current* validators in each semantic mode: model behaviour, latency and tokens are from the live run, validation outcomes are recomputed (a replay, not a new model run) |
| `replay_claims.json` | every claim of every recorded answer with the lexical / rules / NLI verdicts, for manual review |
| `independent/` | the same set of files for the 30 model-generated questions (`data/eval/rag/independent_gemini.yaml`) |
| `semantic_grounding.json` | the synthetic adversarial benchmark: per-category and per-system results on dev and test halves, the dev calibration grid, NLI latency and memory (`data/eval/rag/grounding_adversarial.jsonl`, sha256 in the file) |

Model: `gemini-3.1-flash-lite` on the Google AI Studio free tier through the OpenAI-compatible endpoint. Counted requests:
30 (agent-authored set, including a 1-token probe per run) + 36 (independent set) + 7 (question generation).
