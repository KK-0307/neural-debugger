# Evaluation: dataset, methodology and BM25 baseline (Milestone 2)

Status: **only the BM25 baseline (plus a random floor) has been measured.** The pretrained-embedding
baseline, the custom transformer and hybrid retrieval are Milestones 3-5; their rows do not exist
yet and no claim about them is made anywhere.

Everything here is reproducible with:

```bash
make dataset            # YAML sources -> data/eval/queries.jsonl + manifest, runs integrity/leakage checks
make eval-bm25          # re-score the frozen BM25 baseline on train+val (test is protected)
make eval-ablation      # dev-only configuration study
uv run python scripts/render_eval_tables.py   # the tables below
```

## 1. What is being evaluated

Given a developer query (an error message or a how-to question) and the pinned corpus
`nd-corpus-v1` (4,995 chunks / 4,154 sections, see [data-pipeline.md](data-pipeline.md)), return the
documentation **sections** that answer it. Results are collapsed to one hit per section (best chunk
represents it), and the ranking is scored at section level against the labeled relevant sections.
Section granularity keeps labels valid when chunking parameters change.

## 2. The labeled dataset (`nd-eval-v1`): how labels were obtained

203 queries; 170 of them (84 per cent) are *authored by the project's AI assistant* and **none has been
reviewed by a human yet**. Treat the labels as careful but unverified judgments, not ground truth.

| Slice | n | Query text comes from | Relevant section(s) come from | Provenance in the data |
|---|---:|---|---|---|
| `error_message` | 70 | Written by the assistant to look like real tracebacks / error lines (messages recalled from the frameworks, not copied from the docs) | Chosen by the assistant after reading the candidate sections (`authored_relevant`), then extended by pooled review (below) | `claude_authored` / `llm_authored_unreviewed` |
| `question` | 100 | Written by the assistant as natural developer questions | same | `claude_authored` / `llm_authored_unreviewed` |
| `official_faq` | 33 | **Verbatim FAQ question headings written by the Python / PyTorch maintainers** (`scripts/make_faq_queries.py`) | The section that answers that question in the official FAQ | `official_faq_heading` / `official_source` |

**Pooled relevance review.** For each of the 170 authored queries, the top-8 BM25 sections that were
not already labeled were read and judged (`scripts/pool_candidates.py`; judgments in
`data/eval/pooled_judgments.yaml`): 109 extra sections were marked relevant across 73 queries, all
others judged not relevant. The FAQ slice is deliberately *not* pooled, so its labels stay purely
official. Relevance is binary and judged strictly ("would a developer with this query find the
answer or key diagnostic information here?").

What this does and does not establish:

* Every labeled section exists in the corpus and has a verified citation URL (checked by tests).
* The `official_faq` slice has independent human-written provenance - but it is lexically trivial:
  the question *is* the indexed section heading (33/33 queries appear verbatim in the index text).
* The authored slices were written by one annotator (an LLM) who had read the target sections, so
  query wording may be biased toward the documentation's vocabulary, and no inter-annotator
  agreement exists. **The first thing to do before publishing numbers externally is human review**:
  `human_reviewed_queries` is 0 in `data/manifests/eval_dataset_manifest.json`.
* Pooling used BM25 only. That inflates BM25 (its own top results were what got judged) and will
  *under-count* later systems that retrieve relevant sections BM25 never surfaced. Milestone 4 must
  re-pool with the neural systems before comparing them.
* Synthetic training queries (Milestone 4) will not be mixed into this set.

### Label review (agent second pass; not human verification)

Before freezing the baseline, 102 judgments over 38 train/val queries were re-read against the
section text (`data/eval/label_review.yaml`). Selection prioritised queries with at least 4 relevant
sections, BM25 failures, pooled additions, and random authored queries. **The reviewer is the same
assistant that authored the labels, so this is a consistency check, not independent verification,
and the file says so** (`independent: false`, `human_verified: false`; a test enforces both).
Test-split labels were deliberately not reviewed in this pass.

| Verdict | Judgments | Meaning |
|---|---:|---|
| confirmed | 71 | the section directly answers the query / holds the key diagnostic information |
| marginal | 23 | related background; a stricter annotator might not label it relevant |
| uncertain | 6 | could go either way |
| weak | 2 | probably should not be labeled (stub or tangential) |

Six queries carry an uncertain or weak judgment and are flagged: `fa007`, `fa049`, `fa053`, `pt004`,
`pt020`, `py036`. **Labels were not edited** (the baseline is frozen); the flags are used for
sensitivity analysis (results with and without flagged queries). Marginal judgments mostly come from
pooled review, so expect pooled relevance to be somewhat generous.

### Freeze and test-set protocol

* `data/eval/baseline_freeze.json` records SHA-256 hashes of the corpus, labeled queries, split
  assignment, review file, BM25 configuration and BM25 result files; `make freeze-verify` (and a
  test) fails if any of them drifts. New work adds new files; it does not edit these.
* The held-out test split is opt-in: `evaluate --splits ... test` is refused unless `--final-test` is
  given, and each allowed access is appended to `data/eval/results/test_access_log.jsonl`
  (`ml/evaluation/protocol.py`). The Milestone 2 BM25 test run is recorded there as a retroactive
  entry. All model development uses `train` and `val` only.

### Splits and leakage

* Queries are grouped (identical text or token-set Jaccard >= 0.6 = same group), and a group is
  never split. A group's partition is `sha256(salt + smallest member hash) mod 100`:
  <60 train, <80 val, else test. It is deterministic, order-independent, and adding queries never
  moves existing ones (tested).
* Result: train 122 / val 39 / test 42 (authored-only: 103 / 35 / 32). Sizes are approximate
  because the rule hashes groups rather than forcing exact proportions.
* `make dataset` / the test-suite check: no near-duplicate or identical queries across splits; the
  stored split equals the recomputed one; no empty split; all labeled sections exist; every query
  has at least one authored relevant section. All pass.
* Informational: 17.1 per cent of the test split's relevant sections are also relevant to some train
  query (so a trained model could partly memorise popular sections; a stricter page-disjoint split
  was not used because the deployed system searches the same fixed corpus it is trained on).

## 3. Metrics

Binary relevance over ranked, unique sections. Implemented in
`src/neural_debugger/ml/evaluation/metrics.py`; unit-tested with hand-computed values and
cross-checked against scikit-learn's `ndcg_score` on random cases.

* `Recall@k` = |relevant in top-k| / |relevant| (with several relevant sections Recall@1 is at most
  1/|relevant|; `hit@k` in the JSON files is the any-relevant variant).
* `MRR` = 1 / rank of the first relevant section within the 100 retrieved (0 if none).
* `NDCG@10` = DCG@10 / ideal DCG@10, gain 1, discount 1/log2(rank+1).
* Intervals are 95 per cent percentile bootstrap over queries (2,000 resamples, fixed seed).
  With 32-42 test queries they are wide; differences smaller than the intervals are not evidence.

## 4. Experiment 1 (partial): lexical baseline and sanity floor

BM25 (Okapi, Lucene-style IDF) with a code-aware tokenizer (whole identifiers such as
`torch.nn.module` plus their sub-words; small stopword list; no stemming), `k1 = 1.2`, `b = 0.75`.
This configuration was fixed before any evaluation. A **dev-only** study (train + val; test was not
touched) then checked whether any variant is better; none is (section 6), so the default stands.
The test split was first scored after that, for this single configuration (later re-runs only
changed how results are reported, not the configuration or the data).

**Hardware / workload:** Apple M4 Max (arm64), macOS 26.5, Python 3.12.15, NumPy 2.5.4; one process, one query at a time, index built in memory from `chunks.jsonl`.

### Headline: authored queries (error messages + questions)

| | n | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| test | 32 | 0.329 [0.21, 0.45] | 0.567 [0.42, 0.71] | 0.692 [0.56, 0.82] | 0.666 [0.52, 0.80] | 0.597 [0.47, 0.72] |
| train | 103 | 0.236 [0.18, 0.30] | 0.569 [0.49, 0.65] | 0.686 [0.62, 0.76] | 0.616 [0.54, 0.69] | 0.569 [0.50, 0.64] |
| val | 35 | 0.313 [0.21, 0.43] | 0.628 [0.51, 0.74] | 0.729 [0.61, 0.84] | 0.698 [0.56, 0.82] | 0.619 [0.51, 0.73] |

### Including the official-FAQ slice (inflated by heading overlap)

| | n | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| test | 42 | 0.465 [0.34, 0.60] | 0.670 [0.55, 0.79] | 0.765 [0.66, 0.87] | 0.734 [0.62, 0.84] | 0.684 [0.58, 0.79] |
| train | 122 | 0.347 [0.28, 0.42] | 0.636 [0.56, 0.70] | 0.735 [0.67, 0.80] | 0.672 [0.60, 0.74] | 0.633 [0.57, 0.70] |
| val | 39 | 0.383 [0.27, 0.51] | 0.666 [0.56, 0.77] | 0.756 [0.65, 0.86] | 0.729 [0.61, 0.84] | 0.658 [0.55, 0.76] |

### Where it works and where it does not

By slice (all splits pooled):

| | n | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| error_message | 70 | 0.281 [0.20, 0.37] | 0.564 [0.48, 0.66] | 0.689 [0.60, 0.78] | 0.630 [0.53, 0.73] | 0.573 [0.49, 0.66] |
| official_faq | 33 | 0.939 [0.85, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.970 [0.92, 1.00] | 0.978 [0.94, 1.00] |
| question | 100 | 0.262 [0.20, 0.32] | 0.593 [0.52, 0.67] | 0.701 [0.63, 0.77] | 0.651 [0.57, 0.73] | 0.593 [0.52, 0.66] |

By framework (all splits pooled):

| | n | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| fastapi | 55 | 0.198 [0.13, 0.27] | 0.541 [0.44, 0.64] | 0.673 [0.57, 0.77] | 0.629 [0.52, 0.73] | 0.559 [0.47, 0.65] |
| python | 88 | 0.475 [0.39, 0.57] | 0.740 [0.67, 0.81] | 0.807 [0.73, 0.87] | 0.729 [0.65, 0.81] | 0.712 [0.64, 0.78] |
| pytorch | 60 | 0.402 [0.30, 0.51] | 0.615 [0.52, 0.71] | 0.722 [0.63, 0.81] | 0.708 [0.60, 0.81] | 0.637 [0.54, 0.73] |

Test split by slice (n is tiny; shown for transparency):

| | n | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| error_message | 16 | 0.273 [0.14, 0.42] | 0.560 [0.38, 0.74] | 0.748 [0.58, 0.90] | 0.669 [0.47, 0.87] | 0.614 [0.45, 0.77] |
| official_faq | 10 | 0.900 [0.70, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.950 [0.85, 1.00] | 0.963 [0.89, 1.00] |
| question | 16 | 0.385 [0.20, 0.58] | 0.573 [0.36, 0.77] | 0.635 [0.44, 0.82] | 0.663 [0.47, 0.85] | 0.581 [0.40, 0.76] |

### Reference points (test split, authored only; intervals omitted)

| | n | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| random (seed 0) | 32 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 |
| BM25 | 32 | 0.329 | 0.567 | 0.692 | 0.666 | 0.597 |
| BM25 + framework filter | 32 | 0.340 | 0.577 | 0.733 | 0.696 | 0.622 |

* The random floor is ~0 (4,154 sections), so the metrics are not trivially satisfiable.
* An *oracle* framework filter (restricting search to the query's framework) adds a few points;
  that is the upper bound for what the UI's framework selector buys BM25.

### Cost

- index build: 0.703 s for 4995 chunks
- per-query search (test, n=42): mean 0.753 ms, p50 0.766 ms, p95 0.943 ms, max 0.99 ms

## 5. Failure analysis (descriptive; nothing was tuned from it)

27 of 170 authored queries (16 per cent) have no relevant section in BM25's top 10. Patterns:

1. **Error text vs. explanatory docs.** The traceback wording is absent from the page that explains
   it: `mat1 and mat2 shapes cannot be multiplied` -> `nn.Linear` / `matmul`; `IndexError: Target 10 is out of bounds`
   -> `CrossEntropyLoss`; `not enough values to unpack` -> *tuples and sequences*. Lexical match
   latches onto unrelated sections that contain a few of the same tokens.
2. **Synonym gaps in natural questions.** `how does adding tensors with different shapes work` vs.
   *broadcasting*; `count how many times each item appears` vs. `collections.Counter`;
   `make an enum of named constants` vs. `enum.Enum`.
3. **Cross-framework confusion.** FastAPI queries pull Python-stdlib sections that share error
   words (`RuntimeError: Directory 'static' does not exist` -> Python's *exception chaining*).
   This is why the framework filter helps.
4. **Label incompleteness.** Some "misses" rank a genuinely relevant, unlabeled section first
   (e.g. a Python asyncio page for a FastAPI coroutine warning). The labels are cross-checked for
   BM25's top 8 only; they will be incomplete for other systems.

These are the gaps that dense retrieval and hybrid fusion (Milestones 4-5) should be tested against.

## 6. Dev-set configuration study (train + val, n=161; test untouched)

Paired bootstrap on per-query NDCG@10 against the default:

| configuration | R@1 | R@10 | MRR | NDCG@10 | paired ΔNDCG@10 vs default [95% CI] |
|---|---:|---:|---:|---:|---:|
| plain words (baseline) | 0.357 | 0.730 | 0.696 | 0.634 | -0.005 [-0.027, +0.018] |
| plain + stopwords kept | 0.343 | 0.720 | 0.674 | 0.621 | -0.018 [-0.042, +0.007] |
| identifiers (default) | 0.356 | 0.740 | 0.686 | 0.639 | +0.000 [+0.000, +0.000] |
| identifiers, parts only | 0.356 | 0.739 | 0.692 | 0.642 | +0.003 [-0.011, +0.017] |
| identifiers + stopwords kept | 0.349 | 0.736 | 0.674 | 0.632 | -0.007 [-0.021, +0.007] |
| identifiers + S-stemmer | 0.345 | 0.736 | 0.658 | 0.625 | -0.014 [-0.029, -0.000] |
| k1=0.9 b=0.5 | 0.353 | 0.732 | 0.681 | 0.634 | -0.005 [-0.016, +0.006] |
| k1=0.9 b=0.75 | 0.351 | 0.737 | 0.680 | 0.637 | -0.002 [-0.009, +0.004] |
| k1=0.9 b=0.9 | 0.357 | 0.740 | 0.681 | 0.636 | -0.003 [-0.009, +0.003] |
| k1=1.2 b=0.5 | 0.358 | 0.725 | 0.690 | 0.636 | -0.003 [-0.012, +0.007] |
| k1=1.2 b=0.75 | 0.356 | 0.740 | 0.686 | 0.639 | +0.000 [+0.000, +0.000] |
| k1=1.2 b=0.9 | 0.361 | 0.738 | 0.688 | 0.637 | -0.001 [-0.008, +0.005] |
| k1=1.5 b=0.5 | 0.353 | 0.725 | 0.682 | 0.632 | -0.007 [-0.017, +0.004] |
| k1=1.5 b=0.75 | 0.362 | 0.735 | 0.694 | 0.638 | -0.001 [-0.007, +0.006] |
| k1=1.5 b=0.9 | 0.358 | 0.734 | 0.687 | 0.635 | -0.004 [-0.015, +0.007] |
| k1=2.0 b=0.5 | 0.358 | 0.721 | 0.686 | 0.630 | -0.009 [-0.021, +0.004] |
| k1=2.0 b=0.75 | 0.347 | 0.730 | 0.678 | 0.627 | -0.011 [-0.024, +0.000] |
| k1=2.0 b=0.9 | 0.343 | 0.726 | 0.675 | 0.621 | -0.018 [-0.032, -0.003] |

Findings: **no variant is significantly better than the default**; stopword removal helps slightly
(not significantly); the S-stemmer and `k1 = 2.0, b = 0.9` are slightly worse. In particular the
identifier-aware tokenizer's advantage over plain word tokens (+0.005 NDCG@10, CI [-0.027, +0.018])
is **not demonstrated** on this data, so no claim that it improves retrieval is made.

## 7. Limitations and threats to validity

1. Labels are LLM-authored and unreviewed by a human (see section 2). Human review is the main
   outstanding quality step.
2. Pooling from BM25 alone biases the judged set toward BM25 and will penalise other systems.
3. Small samples: test has 32 authored / 42 total queries; intervals are +/-0.1-0.15. Do not read
   rankings between close systems from this set alone; pool train/val/test for paired comparisons
   once the neural model's training data is separate.
4. Authored queries were written with the target section in view, so lexical overlap is probably
   higher than for real users; the absolute numbers are likely optimistic for real traffic.
5. Section-level binary relevance ignores partial relevance and answer position inside long sections.
6. Corpus is small (207 pages) and pinned; results do not transfer to other versions or to
   unindexed questions. There are no real-user queries yet (Milestone 8 beta).
7. The FAQ slice is near-trivial for any system that indexes headings; always report it separately.
8. Latency is measured in-process on one laptop (Apple M4 Max), with no warm-up phase;
   it says nothing about a deployed service.
