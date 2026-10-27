# Commit hash map (history rewrite on 2026-10-27)

Before the repository was made public, every commit's author and committer e-mail was replaced with the GitHub no-reply address.
The file contents of every commit are unchanged, but that changes every commit hash. The write-ups under `docs/` and `PROGRESS.md`
were written before the rewrite and cite the **old** short hashes; this table maps them to the current ones (same order, same
commit subjects). The frozen evaluation files are hash-checked by content (`make freeze-verify`), not by commit hash, so they are
unaffected.

| old | current | subject |
|---|---|---|
| `8f898a9` | `4a3a1d3` | Milestones 1-2: foundation, documentation corpus, BM25 baseline |
| `82566f2` | `68d33d0` | Milestone 3: neural retrieval (dense, hybrid RRF, rerank) with expanded judgments |
| `d6b3546` | `409dd99` | CI: generate Next route types before typecheck (fresh clones have no .next/types) |
| `e37540c` | `299191a` | Frontend: track public/ so the Docker runner stage can copy it |
| `aa1b0b2` | `dee9a1b` | Milestone 4: from-scratch transformer dual encoder, InfoNCE training, evaluation |
| `bd2cb61` | `c7b09d6` | Milestone 5: RAG with validated citations, provider interface, abstention and injection defences |
| `09f55e8` | `6131bc9` | PROGRESS: record green CI for Milestone 5 |
| `a388397` | `e8f8ed6` | Milestone 5.1: live-LLM verification, semantic grounding, Docker retrieval image |
| `d148b6d` | `6ba2b3f` | CI: make the image secret checks fail explicitly (set -e ignores a failing '! pipeline') |
| `91521dc` | `cc35db6` | Docs: record CI-verified Docker results (image 2,766 MiB, smoke tests green) |
| `35087b0` | `0975332` | Milestone 5.2 (part 1): coverage diagnosis, supplement corpus overlay, traceback API hints, decoupled evidence profile |
| `2619602` | `de00e99` | Milestone 5.2 (part 2) + Milestone 6: held-out evaluation, feedback API, the Neural Debugger web app |
| `339bfe9` | `dcbe75f` | Milestone 5.3 (part 1): length-robust evidence gate, long-trace query fix, cross-framework citation rule, new datasets |
| `377c1ec` | `9d39bbe` | Milestone 5.3 (part 2): reliability write-up, fresh final evaluation, read-only public demo (not deployed) |
| `cbde943` | `42cdcef` | CI: measure whether the backend container fits a 512 MB memory limit (informational) |
| `1d84a59` | `712e893` | Deployment: one-container full-stack image (FastAPI + exported Next.js), public-mode UI, external smoke test, Render blueprint |
| `e6deb99` | `6d35a4a` | Release docs: README rewrite, deployment guide, portfolio notes, MIT license; background retriever warm-up |
| `94920e8` | `4055a2d` | PROGRESS and counts: release milestone |
