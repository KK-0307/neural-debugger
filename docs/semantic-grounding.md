# Semantic grounding: catching claims whose meaning differs from the cited text (Milestone 5.1)

Milestone 5 validated that a citation *exists* (validity) and that the claim shares *vocabulary* with the cited
passage (lexical support). It could not see that "the default timeout is **60** seconds" contradicts a passage
saying 30, or that "tensors **must not** be on the same device" negates the passage: its own fault-injection test
showed 132 of 132 inverted-meaning claims surviving. This stage (`src/neural_debugger/rag/semantic.py`) adds
deterministic checks and an optional NLI model on top of the lexical judge.

Everything below is **automated or heuristic judgement. Nothing was labelled by a human**, and the model-based
parts are not ground truth.

## 1. What it checks

For each claim, against the passages it cites (each sentence of a multi-sentence claim separately):

| signal | what it needs | verdict | default policy |
|---|---|---|---|
| claim is a verbatim quote of the cited text | nothing | `supported` | kept |
| a **number** in the claim differs from the matching passage sentences | rules | `contradicted` | **removed** |
| an API **signature** `f(a, b=1)` differs from the documented one (order, names, defaults) | rules | `contradicted` | **removed** |
| **negation polarity** differs from the matching sentence | rules | `disputed` | kept, flagged, confidence `low` |
| the NLI model says *contradiction* for a topically matched sentence (and no other sentence entails the claim) | NLI | `disputed` | kept, flagged, confidence `low` |
| an **identifier** / number in the claim appears nowhere in the cited text | rules | `unverified` | kept, flagged, downgraded to `partial` |
| nothing contradicts it, but nothing confirms it either | - | `unconfirmed` | unchanged (neutral) |
| an NLI model entails it | NLI | `supported` | kept |

Two things were decided from measurements, not intuition (section 4):

* **Only hard evidence removes a claim.** A changed number or signature is removed. Negation and NLI contradictions are
  *flags*, because on real model output they fired on correct claims (section 4). `ND_RAG_DROP_DISPUTED=true`
  (`PipelineConfig.drop_disputed`) removes them as well.
* **"Could not confirm" is neutral, not a warning.** A 283 MB NLI model entailed only 2 of 41 real paraphrased claims;
  flagging the other 39 would carry no information. `unverified` is therefore reserved for specific findings
  (an identifier or number that is absent from the cited text).

Identifiers and bare numbers the **user typed** are not treated as fabricated when an answer repeats them
(`http://localhost:3000` in a CORS question). Instruction-like sentences in the question are excluded from that trusted
text, so an injected URL or identifier gains nothing from it. Loopback URLs the user typed are the only URLs the
cleaner lets through besides documentation sources.

## 2. Modes

| `ND_RAG_SEMANTIC_CHECK` | cost | use |
|---|---|---|
| `off` | none | the `PipelineConfig` library default; the Milestone 5 evaluation harness uses it (4 of its 146 numbers moved later because of the user-typed-identifier exemption, see section 5) |
| `rules` | ~0.2 ms per claim, no extra memory | **deployment default** |
| `nli` | `cross-encoder/nli-deberta-v3-xsmall` (Apache-2.0, 283 MB, pinned revision `a1508764`): ~37 ms p50 / 73 ms p95 per claim on this Mac's CPU, **+520 MB peak memory**, 6 s load | optional; not baked into the Docker image |

## 3. The adversarial benchmark (synthetic, agent-authored)

`python -m neural_debugger.rag.grounding_eval` builds, deterministically (seed 20261026, id-hash dev/test split),
1,066 items from real corpus sections: `data/eval/rag/grounding_adversarial.jsonl`
(sha256 `19c53022151c5da0...`). Supported claims are verbatim quotes, clauses, two-sentence quotes, signature quotes
and two light **paraphrases** (reorder "If X, Y" -> "Y if X"; prefix "According to the documentation"). Defective claims
are programmatic mutations of a supported sentence: negation, a changed number, a changed signature (swap / rename /
changed default), a fabricated identifier (`detail` -> `detail_async`, guaranteed absent from the corpus), an
antonym swap (curated list), and an appended causal clause taken from an unrelated passage.

Labels are correct by construction, but this is **not** a sample of how an LLM actually errs: mutations are minimal
edits, supported claims are mostly verbatim, the "causal" additions are topically unrelated, and the set skews to
FastAPI (563 of 1,066 items, 288 PyTorch, 215 Python, although the corpus is 3/4 Python; the likely cause is that the prose-sentence filter keeps more FastAPI sentences, which I did not verify). **Thresholds were chosen on the dev half only**
(rule: lowest silent acceptance of defective claims subject to <= 5% rejection of supported ones; all 48 grid points met
the budget), and the numbers below are the **test half** (518 items).

| system (test half) | defective claims silently accepted | defective rejected | supported quotes accepted / rejected | supported paraphrases accepted / flagged / rejected | ms per claim p50 / p95 |
|---|---:|---:|---:|---:|---:|
| lexical only (Milestone 5) | **52.0%** | 24.0% | 100.0% / 0.0% | 100.0% / 0.0% / 0.0% | 0.0 / 0.1 |
| + rules | **21.3%** | 42.9% | 97.7% / 0.6% | 100.0% / 0.0% / 0.0% | 0.2 / 0.3 |
| + NLI only | **3.9%** | 24.0% | 98.9% / 0.0% | 98.9% / 1.1% / 0.0% | 34.9 / 73.5 |
| + rules + NLI | **2.7%** | 42.9% | 97.2% / 0.6% | 98.9% / 1.1% / 0.0% | 36.0 / 72.5 |

Per category, test half (accepted silently / flagged / rejected), n in brackets:

| category | n | lexical | + rules | + NLI only | + rules + NLI |
|---|---:|---|---|---|---|
| supported_verbatim | 43 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| supported_clause | 43 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| supported_two_sentences | 42 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| supported_signature | 48 | 100 / 0 / 0 | 92 / 6 / 2 | 96 / 4 / 0 | 90 / 8 / 2 |
| supported_reordered | 43 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| supported_prefixed | 45 | 100 / 0 / 0 | 100 / 0 / 0 | 98 / 2 / 0 | 98 / 2 / 0 |
| negation | 50 | 100 / 0 / 0 | 8 / 92 / 0 | 2 / 98 / 0 | 2 / 98 / 0 |
| wrong_number | 33 | 91 / 6 / 3 | 3 / 0 / 97 | 0 / 97 / 3 | 0 / 3 / 97 |
| wrong_signature | 46 | 6 / 30 / 63 | 0 / 0 / 100 | 6 / 30 / 63 | 0 / 0 / 100 |
| fabricated_identifier | 37 | 0 / 57 / 43 | 0 / 57 / 43 | 0 / 57 / 43 | 0 / 57 / 43 |
| antonym_swap | 45 | 98 / 2 / 0 | 98 / 2 / 0 | 2 / 98 / 0 | 2 / 98 / 0 |
| unsupported_causal | 43 | 12 / 54 / 35 | 12 / 54 / 35 | 12 / 54 / 35 | 12 / 54 / 35 |

Reading it:

* The lexical judge alone silently accepts **52%** of defective claims, including 100% of negations and 98% of
  antonym swaps. Rules alone: 21%. Rules + NLI: **2.7%**, but "not silently accepted" mostly means *flagged and kept*
  (negations, antonyms) or *removed* (numbers, signatures), per the policy above.
* **What each stage cannot do:** rules cannot see antonyms (98% pass); NLI alone catches few changed numbers as
  *rejections* (it flags them); nothing here detects a fabricated identifier any better than the lexical judge
  (57% flagged, 43% rejected in every system, because the lexical identifier check already catches them); causal
  additions that reuse passage vocabulary still pass 12% of the time.
* Supported **paraphrases** are never penalised in the table because "unconfirmed" is neutral. That is a deliberate
  design choice, not evidence that the NLI model understands paraphrase (section 4 shows it does not).

## 4. The check that mattered: real model output

`python -m neural_debugger.rag.live_replay` re-judges the recorded replies of the live Gemini run (60 claims in 21
answers) offline. The first design, which treated every NLI contradiction as grounds for removal and every
non-confirmation as a flag, was measured here and **rejected**:

| first design on 41 real claims | result |
|---|---|
| claims marked `unverified` (not confirmed) | 35-39 of 41 (85-95%): a flag on almost everything |
| claims marked `contradicted` | 5, all five correct claims (e.g. "Await the coroutine directly using the 'await' keyword", "FastAPI requires the 'python-multipart' package ..." at NLI contradiction 0.99) |
| negation-polarity rule | fired on "schedule the coroutine ... using asyncio.create_task()" because the documentation sentence it matched describes the error ("was *never* awaited") |

Changes made because of this, all kept and tested: negation and NLI contradiction became *flags*, not deletions;
generic non-confirmation became neutral; dev-calibrated thresholds became the shipped defaults. Re-run with the
final code on all real claims (60 from the 21 agent-authored questions, 38 from the 30 independent ones):

| real claims | `rules` (default) | `nli` |
|---|---|---|
| 60 (agent-authored questions) | 3 supported, 57 unconfirmed, 0 disputed, 0 unverified | 5 supported, 54 unconfirmed, **1 disputed**, 0 unverified |
| 38 (independent questions) | 1 supported, 32 unconfirmed, **2 disputed**, **3 unverified** | 4 supported, 29 unconfirmed, **2 disputed**, **3 unverified** |

Agent review of every flagged claim: the 2-3 `disputed` claims are false alarms on correct claims ("Move all tensors to the
same device using ..."; two FastAPI claims advising `Form` parameters instead of a JSON body, which the docs recommend), and
the 3 `unverified` claims are fair observations (an illustrative `class Bag[T]:` and `python --version` that the cited text
does not contain, and a version number absent from the cited passage). Nothing was removed as `contradicted`.

What this does and does not show: **false disputes on real, correct claims are low (0-3 in 98) but the real sample contains
no known defective claim**, so the real-world recall of the stage is **unmeasured**. The synthetic table is the only recall
estimate, and section 3 lists why it flatters the stage. A disputed claim lowers the answer's confidence to `low` even when
the claim is correct (the multipart answer above), which is a real usability cost of the flag policy.

## 5. Bugs the live data exposed in the older lexical judge (fixed)

1. `e.g.` / `i.e.` were parsed as dotted API names, so correct claims containing them were rejected for
   "missing identifiers" (2 of 6 lexical rejections on the live run).
2. A dotted name (`torch.cuda.empty_cache()`) was required to appear verbatim, but API reference pages list the bare
   function name under its module; the final segment is now accepted when it appears as a whole word. A
   fabricated final segment (`empty_cache_async`) still fails.
3. A value the user typed (`http://localhost:3000`) was treated as a fabricated identifier, and the URL cleaner replaced
   it with `[link removed]` (and swallowed the closing backtick). Both now trust the user's own non-injection text; only
   loopback URLs are let through.

After 1-3 the lexical judge rejected 4 of 60 real claims instead of 9. Frozen Milestone 5 results were re-run with the
fixed code: all 146 non-timing numbers were identical at that point. (Correction: the user-typed-identifier exemption of item 3 was added after this check; with it, 4 of the 146 numbers differ, all confidence counts on one train answer, low -> medium. Found and attributed in Milestone 5.2.)

## 6. Limitations

* No human labels; the benchmark is synthetic; thresholds come from a single dev half of one generated set.
* Recall on real hallucinations is unmeasured (no known defective claim in the real sample).
* English only. The antonym list is small and curated. NLI works on sentence windows, so a claim that needs two
  distant passages to be entailed is `unconfirmed`, not `supported`.
* `disputed` claims are shown, not removed, by default; a user who ignores the flag still sees them.
