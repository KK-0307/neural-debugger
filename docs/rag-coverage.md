# Answer coverage: where in-scope questions were lost, and what helped (Milestone 5.2)

Goal: answer more of the unfamiliar, real-world questions the system had been declining, **without** weakening citation
integrity, groundedness or abstention. Everything below is **agent-authored, model-generated or agent-reviewed; nothing was
reviewed by a human.** The only live model is Gemini 3.1 Flash-Lite on the free tier.

## 1. Result in one table (held-out, never used for tuning)

48 questions written by Gemini (which never saw the corpus or the system): 36 in scope (half with a pasted traceback or server
log) and 12 out of scope (Kubernetes, Rust, Django, NumPy...). Generated once, hashed, and frozen **before any change**;
`data/eval/rag/heldout_gemini.yaml`, sha256 `4b829806...`. BEFORE = commit `91521dc`, AFTER = `35087b0`; same model, caps and pacing.

| measure | before | after |
|---|---:|---:|
| in-scope questions answered | 11 / 36 (30.6%; 95% CI 18-47%) | **14 / 36 (38.9%; 95% CI 25-55%)** |
| ... Python / PyTorch / FastAPI | 2/12 / 3/12 / 6/12 | 5/12 / 2/12 / 7/12 |
| out-of-scope questions that abstained | 12 / 12 | 12 / 12 |
| invalid citation ids | 0 of 38 claims | 0 of 56 claims |
| claims removed by the lexical support check | 5 of 38 | 9 of 56 |
| claims flagged by the semantic check (disputed / unverified) | 0 / 4 | 2 / 2 |
| answers judged good / generic / weak (agent review) | 8 / 1 / 2 (of 11) | 10 / 1 / 3 (of 14) |
| in-scope abstentions judged appropriate / avoidable / borderline (agent review) | not reviewed | 8 / 6 / 8 (of 22) |
| generation latency p50 / p95 | 10.0 s / 21.5 s | 8.3 s / 21.6 s |
| input tokens, whole run | 51,523 | 72,932 (+42%) |

**How much to believe it.** Four questions were gained (hp8, hp10, hp12, hf12) and one lost (ht3): net +3. An exact sign test on
those five changes gives p = 0.375 and the intervals overlap heavily, so this is a **directional improvement, not a demonstrated
one**. PyTorch got slightly worse (3 -> 2). The three Python 3.12 gains are exactly the kind of question the supplement pages
(section 4) were added for. The lost answer (ht3) was a good one before (`detect_anomaly` for in-place errors) and the model
declined after it was shown 7 sections instead of 4, even though the relevant text was visible in both runs: extra evidence may
distract the model, but at n = 36 that cannot be separated from ordinary model variability.

## 2. Diagnosis (development questions only)

Of the 48 development questions (the 21 agent-authored and 30 independent ones from Milestone 5.1, minus injection tests), 13 have answer-bearing text
somewhere in the indexed documentation (annotated by searching the corpus: `data/eval/rag/coverage_dev.yaml`). The pipeline up to
the gate (`python -m neural_debugger.rag.coverage_eval`) says where each was lost:

| stage that lost the answer | before | after |
|---|---:|---:|
| reachable (the model would be shown the answer-bearing text) | 1 | **7** |
| corpus gap (section not in the index) | 5 | 0 |
| not retrieved (ranking: rank 13-36, beyond `retrieve_k`=12) | 5 | 2 |
| retrieved but not shown (evidence selection) | 2 | 4 (3 of them are former "not retrieved") |
| blocked by the sufficiency gate | **0** | **0** |
| *plus* questions with no answer anywhere in the corpus | 5 | 5 |

So on the development set the three suspects were **not** equally guilty: corpus gaps and ranking dominated, evidence selection
cost two, and **the gate blocked none of the answerable questions**; its six in-scope abstentions were questions with no answer
or a retrieval miss. That is why the gate thresholds were not touched in this pass. (The held-out set later showed the gate
*can* over-abstain on verbose questions: section 6. I did not tune on that.)

## 3. What changed, and what it measured

1. **Supplement corpus (official pages only).** Four pages from the same documentation sets and licences: `library/sys`,
   `library/sqlite3`, `whatsnew/3.12` and `torch.compile` (361 chunks, +7%), each added because a *development* question showed
   its content missing (`sys.setrecursionlimit`, the 3.12 sqlite3 CLI, PEP 684, "Did you mean" for NameError, the `distutils`
   removal). Built with the existing pipeline into `data/corpus_supplement/` with its own manifest and **merged in memory**
   (`load_corpus_overlay`), with its own embeddings in `data/embeddings_rag/`: the frozen base corpus, its labels, manifest and
   embeddings are not modified. Effect: 5 corpus-gap questions became reachable.
2. **Library-frame API hints for tracebacks.** `File ".../torch/nn/functional.py", line 3479, in cross_entropy` now adds
   `torch.nn.functional.cross_entropy` to the retrieval query (innermost 3 library frames; user frames, private modules and
   plumbing like `forward` are skipped), and uvicorn/INFO log lines are dropped from the query. Measured on 10 agent-written
   development tracebacks: gold page in the evidence 9 -> 10, gate abstentions 1 -> 0. Costs: two items slipped a few ranks
   (tb2 2 -> 4, tb5 1 -> 3) but stayed in the evidence. The hints are **excluded from the gate's coverage text**: a first version
   counted a module path the docs never spell out (`asyncio.runners`) as "uncovered" and turned a good match into an abstention.
3. **Evidence profile that does not loosen the gate.** The model is now shown up to 7 sections (max 3 per page, 8,000 characters)
   from 16 candidates, **while the sufficiency gate keeps judging exactly the Milestone 5 evidence** (5 sections, 2 per page, top
   12). Showing more evidence raises the gate's coverage score, and when I measured that directly out-of-scope abstention fell
   from 81% to 69-72% (dev, n = 32) and from 8/9 to 6/9 on the new out-of-scope questions: exactly the weakening of a safeguard
   the brief forbids. Decoupling keeps gate decisions identical by construction (`tests/rag/test_coverage.py` proves the extra
   sections do not change the gate, and that coupling them would). Effect: golds reachable 1 -> 7 together with item 1; cost:
   +42% input tokens. `PipelineConfig()` keeps the Milestone 5 behaviour; the backend default is `rag_profile=coverage`.

### Regression guard (161 development retrieval queries + the unanswerable set; base corpus vs overlay)

| | base (M5) | overlay + coverage profile |
|---|---:|---:|
| NDCG@5 | 0.668 | 0.657 |
| NDCG@10 | 0.712 | 0.705 |
| hit@5 | 0.851 | 0.839 |
| false abstention on train answerables | 3.9% | 4.9% (one more query of 103) |
| unanswerable abstained (all 32 / calibration half / held-out half) | 26/32 (14/16, 12/16) | 26/32 (13/16, 13/16) |

New, unjudged pages can only displace judged ones, so a small NDCG drop is expected; it is reported, not hidden.

## 4. Tried and not adopted (so nobody repeats them)

All measured on the 8 development questions whose gold is in the base corpus plus the 161 retrieval queries
(`scripts/m5_2/`). Gold ranks, in order: `dict.get`, `in-place ops`, `type not subscriptable`, `CUDA_LAUNCH_BLOCKING`, `Tensor.grad
is None`, `422 validation`, `CORS wildcards`, `dependency_overrides`.

| idea | gold ranks | dev NDCG@5 | verdict |
|---|---|---:|---|
| baseline hybrid | 17, 36, 14, 6, 16, 13, 5, 2 | 0.668 | - |
| compound-word normalisation (`inplace` -> `in place`) | 17, 36, 13, 6, 16, 13, 5, 2 | 0.671 | no gain (one rank) |
| IDF-weighted lexical re-score of the top 40, fused by RRF | 20, 31, 3, 9, 20, 12, 3, 2 | 0.655 | mixed, small loss |
| query decomposition (error line + each sentence, RRF) | 22, -, 14, 6, 16, 13, 6, 2 | 0.668 | no gain; one gold dropped out |
| cross-encoder rerank (L6, top 30) | 4, 36, 3, 7, 10, 19, 9, 1 | 0.618 | helps some, costs 0.05 NDCG and ~18x latency (196 vs 11 ms/query); MS-MARCO licence caveat |
| cross-encoder rerank (L12 / RRF variants) | 9-14, 36, 2-4, ... | 0.616-0.676 | no consistent gain |

So there is **no cheap query-side fix for the remaining ranking misses**; those golds simply sit far from verbose questions in
the embedding and BM25 spaces.

## 5. Troubleshooting sources: licences and the decision

* **Official documentation** (Python PSF-2, PyTorch BSD-3, FastAPI MIT): same licences as the base corpus, fetched at 1 request/s
  with `robots.txt` consulted. Ingested (section 3.1).
* **Public GitHub issues: not ingested.** GitHub's Terms of Service say users own what they post, and grant other users a licence
  only to "use, display, perform and reproduce" it *through the Service*. Nothing there licenses redistributing issue text
  inside a dataset or image, and a repository's code licence does not cover its issue threads. I recorded the decision rather
  than risk it; issues could be *linked* by a future feature, not copied.
* **Stack Overflow: not used** (no scraping, per the brief); its content terms were not reviewed.
* Writing troubleshooting notes myself was rejected: it would break the guarantee that every cited sentence comes from official
  documentation.

## 6. Quality review of the held-out run (agent review; no tuning followed)

`data/eval/results/m5_2/heldout_agent_review.yaml` holds every judgement.

* **Answers after** (14): 10 good, 1 generic, 3 weak. The weak ones are worth knowing: **hp7** blames an old FastAPI for
  `ImportError: cannot import name 'Annotated' from 'typing' (unknown location)`, which normally means a local file shadows
  `typing`; **hp8** gives generic steps and no diagnosis; **ht9** (an in-place-operation traceback) cites an unrelated
  forward-mode-AD section because the right section was never retrieved (the same retrieval miss found in the development set),
  and the lexical support check let that weakly related citation through. Citation *validity* is perfect; citation *relevance*
  is not guaranteed by these checks.
* **Abstentions after** (22 in scope): 8 appropriate (third-party libraries, fabricated or internal errors the docs do not
  cover), 8 borderline, and **6 avoidable**: four **pre-LLM gate failures** (hp2 TaskGroup cancelling siblings, hp4 the `distutils`
  replacement, hp5 `Path.walk` and symlinks, ht6 `CUDA_LAUNCH_BLOCKING`), where the retrieved evidence contained the answer but
  term coverage (0.22-0.31) fell under the 0.375 threshold, plus two model-side declines. On verbose, conversational symptom
  questions the gate **does** over-abstain; the development set had not shown it. hp4 is the same content that answered hp10
  (a stack-trace-style question) a few lines earlier.
* **Out of scope**: 12 of 12 abstained (11 by the gate, 1 by the model); no question was answered anyway.

## 7. Safeguards that were preserved

No gate threshold, citation validator, support check, semantic check, prompt-injection control or no-execution guarantee was
weakened or removed; the per-run request cap (<= 70 here), 7 s pacing, free-model allowlist and absence of any fallback model are
unchanged; all keys stay in the environment. The new code paths add one rule: hints and user-typed values never count *for*
or *against* the gate.

## 8. Remaining limitations and what to do next

1. **The gate over-abstains on verbose symptom questions** (4 clear cases on held-out). A filler-aware coverage score is the
   obvious next step, but it must be calibrated on a *new* development set, not on this held-out one, and checked against the
   out-of-scope set.
2. **Retrieval misses remain** for 2 of 13 development golds (ranks 16-36) and for traceback-only questions whose cause is not in
   the error line.
3. **Corpus coverage is narrow**: 28 + 3 Python library modules, the 3.12 "what's new" page, a PyTorch API subset. Many
   questions are about content that is simply not indexed.
4. **Citation relevance** is only as good as the lexical/semantic checks (ht9).
5. n = 36 with one model on one day, free-tier outages (3 of 48 held-out questions needed a rerun after Google 503s), no human
   review.

## 9. Reproduce

```bash
python -m neural_debugger.data.ingest build --config configs/sources_supplement.yaml \
  --out-dir data/corpus_supplement --manifest data/manifests/corpus_supplement_manifest.json
python -m neural_debugger.rag.coverage_eval --corpus-dir data/corpus data/corpus_supplement \
  --embedding-dir data/embeddings_rag --profile coverage --out data/eval/results/m5_2/coverage_dev_after.json
python -m neural_debugger.rag.live_acceptance --questions data/eval/rag/heldout_gemini.yaml --profile coverage \
  --out-dir data/eval/results/m5_2/heldout_after          # <= 70 requests, free tier
make freeze-verify                                          # includes verify-m5_2
```
