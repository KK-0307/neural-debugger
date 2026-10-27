# Evidence-gate reliability and unsupported citations (Milestone 5.3)

Scope: one narrow pass on two problems found in 5.2: (a) the evidence gate refuses verbose questions before the model is called even
though the answer is in the retrieved text, and (b) a citation can be valid (the id exists) yet irrelevant. Nothing was retrained, no
other LLM provider was added. Everything below is **agent-authored, model-generated or agent-reviewed; no human reviewed any of
it.** The only live model is Gemini 3.1 Flash-Lite on the free tier.

## 1. Result in one table (fresh evaluation, never used for tuning)

54 questions written by Gemini (which never saw the corpus or the system): 42 in scope (python 14, pytorch 14, fastapi 14) and 12
out of scope. Generated **before any 5.3 change**, hashed and frozen (`data/eval/rag/final_eval_gemini.yaml`), run once BEFORE
(`2619602`) and once AFTER (`339bfe9`) with the same model, pacing and caps. This is a different set from the 5.2 held-out set,
which was not touched.

| measure | before | after |
|---|---:|---:|
| in-scope questions answered (**answer coverage**) | 12 / 42 (28.6%; Wilson 95% 17-44%) | **16 / 42 (38.1%; 25-53%)** |
| ... Python / PyTorch / FastAPI | 7.1% / 21.4% / 57.1% | 14.3% / 42.9% / 57.1% |
| in-scope questions the gate refused before the model | 21 | 11 |
| in-scope questions the model declined itself | 9 | 15 |
| out-of-scope questions that abstained | 12 / 12 | 12 / 12 |
| answers judged good / generic-or-mixed / weak (**grounded quality**, agent review) | 8 / 4 / 0 (of 12) | 11 / 4 / 1 (of 16) |
| abstentions judged appropriate / borderline / avoidable (agent review, after only) | n/a | 16 / 7 / 3 (of 26) |
| invalid citation ids | 0 of 50 claims | 0 of 67 claims |
| claims removed by the lexical support check | 4 of 50 | 5 of 67 |
| claims removed as contradicted / flagged disputed / flagged unverified | 1 / 1 / 3 | 1 / 1 / 3 |
| generation latency p50 / p95 | 10.6 s / 17.1 s | 7.4 s / 21.4 s |
| input tokens, whole run | 69,633 | 101,525 (+46%) |

**How much to believe it.** Four questions were gained and none lost (exact two-sided sign test p = 0.125), and the intervals
overlap. This is a **directional improvement, not a demonstrated one**; n is small. Coverage is still only 38%: PyTorch and Python
questions are mostly refused, and 15 of the in-scope refusals now come from the *model* ("the documentation does not say"), which
is the intended behaviour when the indexed text does not contain the cause. The after-run has one weak answer (`fy1`: a
`torch.compile` graph-break question answered with CUDA-graph material; flagged partial / low confidence by the pipeline). The
quality columns are my own reading of the outputs (`data/eval/results/m5_3/final_agent_review.yaml`): not independent, not human.
The 7 "borderline" and 3 "avoidable" abstentions are known misses, listed in section 6.

The summary is `data/eval/results/m5_3/final_summary.json`; raw replies are frozen next to it (`final_before/`, `final_after/`).
Google's free tier returned "high demand" 503s on some requests; only those infrastructure errors were re-run (`rerun_infra_errors/`)
and the as-run merge is kept (`live_merged_as_run.json`).

## 2. Diagnosis: why answerable questions were refused

A new **development** set of 48 verbose questions (`data/eval/rag/verbose_dev.yaml`: 36 in scope, 12 out of scope, long
conversational prose, pasted tracebacks and server logs) was written by Gemini from sampled documentation sections, so the gold
section is known by construction. Seeded sampling, hashed prompts, never mixed with the 5.2 held-out set.

`scripts/m5_3/diagnose_gate.py` replays the pipeline up to the gate with no model calls:

| | before |
|---|---:|
| in-scope verbose questions the gate refuses | **18 / 36** |
| ... of which the gold section was retrieved and shown to the model | 14 |
| gold section retrieved at all | 35 / 36 |
| out-of-scope verbose questions the gate refuses | 9 / 12 |

So this was not a retrieval failure: for 14 questions the right text was in the evidence and the gate refused anyway. The cause is
the gate's signal. It measures *lexical coverage*: the IDF-weighted share of the question's words that appear in the evidence.
A long conversational question contains many words that appear nowhere in the documentation ("kills", "roadblocks",
"gracefully", "sticking") and an out-of-vocabulary word gets the **highest** IDF weight, so the longer and chattier the question the
lower the coverage, regardless of how relevant the evidence is (AUC separating answerable from out-of-scope on the verbose set:
coverage 0.82, dense similarity of the top section 0.99; on the old calibration sets both are 0.96-0.97).

## 3. The fix (one change, kept)

The gate now sees the dense retrieval similarity it was already computing for ranking (cosine of the question to the top
`gte-small` section):

`score = lexical_coverage + 0.5 * (dense_top1 - 0.80) / 0.10`, abstain below `0.70`

The weak/ok split (a "weak" case still goes to the model but caps confidence) is still decided by lexical coverage alone (0.65).
When no similarity is available (a retriever without a dense part, or the similarity lookup raised) the gate falls back to the
old lexical 0.375 rule; a similarity failure never fails a request. The deployed profile (`PipelineConfig.coverage_profile()`) turns
it on; `PipelineConfig()` (Milestone 5 behaviour) is unchanged. The trace now reports `similarity` and `score` for the gate.

Constants were chosen on the calibration half of the *development* sets only (train authored questions, verbose dev) and then
checked on all development groups (`scripts/m5_3/gate_regression.py`, `data/eval/results/m5_3/gate_regression.json`):

| development group | n | gate refuses, old | gate refuses, new |
|---|---:|---:|---:|
| answerable: train authored (calibration) | 103 | 5 | 4 |
| answerable: val authored | 35 | 0 | 1 |
| answerable: dev own + independent | 39 | 5 | 2 |
| answerable: tracebacks | 10 | 0 | 0 |
| answerable: verbose (diagnosis set) | 36 | **18** | **1** |
| out of scope: unanswerable, calibration half | 16 | 13 | 14 |
| out of scope: unanswerable, held-out half | 16 | 13 | 16 |
| out of scope: dev own + independent | 9 | 8 | 9 |
| out of scope: verbose | 12 | 9 | 10 |

Out-of-scope abstention went **up** in every group (the dense signal also separates adjacent-framework questions that share
vocabulary), and the one answerable regression is on `val authored`. The verbose dev set with the live model afterwards (not a
fair measure, it was used for tuning): 31 of 36 answered or partly answered, 12 of 12 out-of-scope abstained (10 by the gate, 2 by
the model), 0 invalid citation ids (`verbose_dev_after/`).

## 4. Other demonstrated failures that were fixed

* **Long pasted traces polluted the retrieval query.** The question preparation cut a 60-frame traceback in the middle of a line
  and kept indented frame bodies and log noise as "prose", so frames (file paths, `site-packages`) outweighed the error line.
  It now cuts on line boundaries, skips indented frame lines and the truncation marker, and still finds the exception line
  (`question.py`; tests: 60-frame traceback with log noise, deep recursion, whole-pipeline verbose trace).
* **Valid-but-wrong-framework citation.** On `vd33` (FastAPI memory growth) the model cited PyTorch `DataLoader` documentation.
  The id was real and the words overlapped. New rule (profile default on): if the question is unambiguously PyTorch or FastAPI and
  every citation of a claim is from the *other* of the two (Python docs never count as the other), the claim is treated as
  unsupported and removed. Verified directly: unsupported claims on that question went from 1 to 2. Tests cover both directions
  and the Python-docs exemption.
* **Demo limiter** (section 7 of `docs/demo.md`): `Retry-After` was 61 s for a 60 s window (truncation) and the per-client table
  was unbounded; fixed with a ceiling and an LRU bound.

## 5. Things tried and rejected (negative results kept)

Gate experiments (`scripts/m5_3/gate_experiment.py`; development sets only; threshold re-chosen for each variant by the Milestone 5
rule: highest abstain threshold with at most 5% false abstentions on train authored questions). "OOS" = out-of-scope questions
refused by the gate (16 + 16 unanswerable, 9 dev, 12 verbose); "verbose in-scope" = answerable verbose questions refused (of 36):

| variant | verbose in-scope refused | unanswerable OOS refused (of 16 + 16) | verbose OOS refused (of 12) |
|---|---:|---:|---:|
| Milestone 5 lexical gate (baseline) | 18 | 13 + 13 | 9 |
| ignore common English words missing from the corpus | 14 | 10 + 12 | 9 |
| ... and score only the top-8/10/12/16 highest-IDF terms | 24-28 | 10-11 + 12-13 | 11-12 |
| top-12 IDF, common words kept | 28 | 13 + 13 | 12 |
| ignore common words, IDF squared (threshold 0.275) | 8 | 11 + 12 | 9 |
| **combined with dense similarity (adopted)** | **1** | **14 + 16** | **10** |

Every lexical-only variant that lets verbose questions through also lets out-of-scope questions through (or does not help); only
adding the dense signal improved both at once.

Citation-relevance experiments (150 synthetic pairs, section 6):

| idea | result | decision |
|---|---|---|
| "distinctive term" demotion in the support judge (a claim sharing only generic verbs with its source is unsupported) | demoted 3 of 98 real claims on generic verbs and did **not** catch the target irrelevant-citation case | reverted |
| IDF-weighted coverage with a stricter threshold (0.6-0.8) | rejects adjacent pages (not-rejected 34% -> 14% at 0.6) but also rejects 39-43% of faithful paraphrases; at 0.7-0.8, 62-86% | rejected: trades one error for another |
| restrict to high-IDF terms (>= 3, 4, 5) | same trade-off: fewer adjacent survive, 34-74% of true sources rejected | rejected |
| dense similarity of claim vs cited section | AUC true-vs-adjacent **0.54** (IDF coverage 0.78, current judge 0.74); a true paraphrase and an adjacent page have the same cosine range | rejected |

## 6. Limitations (not fixed)

* **The lexical support judge rejects faithful paraphrases.** On 150 synthetic pairs (a Gemini paraphrase of a claim versus its true
  source section, and versus the dense-nearest different section that lacks most of its distinctive terms;
  `data/eval/rag/citation_relevance_dev.jsonl`, `scripts/m5_3/citation_relevance.py`), the current judge:

  | | dev (n=76) | test (n=74) |
  |---|---:|---:|
  | true source judged supported | 20% | 31% |
  | true source judged **unsupported** (false rejection) | 29% | 18% |
  | adjacent page judged supported | 4% | 3% |
  | adjacent page *not rejected* (supported or partial) | 34% | 36% |

  So roughly a quarter of faithful paraphrases are dropped (the safe direction, and consistent with the lower coverage in the
  table above), while about a third of claims pointing at a topically adjacent page survive as "partial" and get the visible
  "partly supported" badge. No cheap signal I tried separates the two cases; this is a strict `xfail` test
  (`tests/rag/test_reliability.py`) so it cannot be forgotten. Pairs are model-generated, not human-verified.
* **Gate near-threshold misses.** In the agent review of the after-run, the gate's refusals that I judge avoidable (`fp2`, `fp7`)
  or borderline (`fp8`, `ff5`, `ff10`) all scored 0.64-0.68 against the 0.70 threshold. The model's own declines include one
  avoidable (`ff7`) and four borderline (`fy3`, `fy9`, `fy10`, `ff2`). I did not move the threshold after seeing the final set
  (it is spent: further tuning on it would be tuning on the test).
* **PyTorch and Python coverage are still low** (43% and 14% of in-scope questions answered on the final set): the indexed
  documentation does not hold answers for many real questions, and no further corpus pages were added in this pass (scope).
* Sample sizes are small; none of the gains is statistically significant.

## 7. Reproducibility drift (frozen results preserved)

All frozen files (M2-M5.2) verify (`make freeze-verify`; the new `data/eval/m5_3_freeze.json` covers the 5.3 sets, results and the
recorded demo data). Re-running the frozen Milestone 5 *extractive* evaluation with today's code differs in exactly **4 of 146**
non-timing numbers (confidence counts: one train answer now gets "low" instead of "medium"), caused by the 5.1
user-typed-identifier exemption; neither 5.2 nor 5.3 adds any. The M5, M5.1 and M5.2 files themselves were not edited. An earlier
note in `PROGRESS.md` saying "146 of 146 identical" was true when checked and stale after that 5.1 change; it was corrected.

Free-tier model output also varies run to run: a re-run of the same question can flip between answered and declined, which is why
the table reports counts with intervals and why every raw reply is stored.

## 8. Reproduce

```bash
python -m neural_debugger.ml.evaluation.freeze verify-m5_3            # frozen 5.3 evidence intact
python scripts/m5_3/diagnose_gate.py                                  # gate diagnosis, no model calls
python scripts/m5_3/gate_regression.py                                # old vs new gate on all development groups
python scripts/m5_3/citation_relevance.py                             # judge on the 150 synthetic pairs
# live evaluation (needs GEMINI_API_KEY in the git-ignored .env; free tier, 7 s spacing, <= 70 requests)
python -m neural_debugger.rag.live_acceptance --questions data/eval/rag/final_eval_gemini.yaml --profile coverage \
  --out-dir data/eval/results/m5_3/final_after
```

Tests: `tests/rag/test_reliability.py` covers verbose and 60-frame stack traces, the gate on verbose / out-of-scope /
no-similarity / similarity-failure cases, irrelevant-but-valid citations, unsupported claims removed, an all-unsupported answer
abstaining (`no_supported_claims`), invalid ids discarded, the cross-framework rule, and the profile defaults.
