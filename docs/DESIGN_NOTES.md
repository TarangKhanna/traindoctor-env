# Design notes

Short record of the design decisions that shape this repository, distilled from the development log (the log itself is not published). Commit hashes cited in `evidence/RELEASE_MANIFEST.md` refer to that private development history.

## Scope
- Executable failures of small PyTorch/Gloo CPU jobs (2-4 ranks) rather than simulated logs. Two task families: **A** (same timeout, two causes: inconsistent collectives vs a rank stalled before communication) and **B** (restart with checkpoint/sampler inconsistency: skipped samples, duplicated samples, and a healthy control). Every artifact carries `source: executed | simulated`.
- The contribution claimed is the end-to-end task plus an audited verifier. No "first" claims; adjacent work is listed in the README.

## Decisions
- **Execute once, snapshot, replay.** A scenario is run once and its artifacts (per-rank logs and collective trace, exit codes, config, checkpoint manifest) are frozen. Read-only tools serve the snapshot; the first fix action ends the read-only phase and the fix is applied to a copy of the job and re-executed.
- **Recovery is judged on invariants, not on exit status.** Committed-sample accounting per optimizer step against an uninterrupted reference, no sample in two committed updates, none skipped, restored model/optimizer/sampler state equal to what the checkpoint committed, lost work within one checkpoint interval, rank consistency, tamper checks. Bit-identical final parameters satisfy only the numerical sub-check.
- **Verifier ladder.** V0 label correct; V1 adds cited evidence that was observed and satisfies the mechanism's evidence predicate; V2 adds verified repair by re-execution. Components are always reportable separately; claimed-vs-verified is logged for every episode.
- **Nothing inside a job process is trusted.** Sample IDs come from a launcher-owned data server (salted, rank identified by socket); checkpoints are copied by the launcher on commit; evidence is harness-owned instrumentation, not logs the job writes. Untrusted repairs run only in a sandbox (`modal_sandbox`: network-less gVisor Modal Sandbox, invariants recomputed outside). `dev_container` (trusted pinned code in local Docker) is a training-reward view only and is never counted by the strict verifier.
- **Tier A is a projection of Tier B.** Tier A is a single-turn, token-capped bundle (3,072 Qwen3 tokens; raised from 2,048 because the Task B certificate alone needs 2,121-2,432 tokens) so a 4B model can be trained; Tier B is multi-turn with real source edits. Tier A results are not Tier B results.
- **Leak tests are statistical,** not string scans: no public field (or simple function of one, e.g. `seed % k`) may predict the label or the recovering repair above chance. A menu-only predictor initially beat chance; the menu was fixed and the tests were mutation-checked (17/17 mutants caught; same-author validation).
- **Splits are by seed, not by mechanism.** DEV and TRAIN seed ranges are in `evidence/archive/notes/splits.md`. The sealed TEST seed pool is deliberately not published; only the SHA-256 of the seed list is recorded in `evidence/prereg/test_protocol.md`.
- **Pre-registration.** Power analysis, minimum meaningful difference (7.2 points), the outcome classes and the TEST protocol were written before the runs they govern (`evidence/prereg/`).
- **Core stays plain Python.** `traindoctor_env/`, `jobs/`, `audit/` import no OpenEnv, Inspect or MLX; those belong to adapters and RL code.

## Future work
- A scenario family where several plausible repairs differ in whether training recovers; today each menu has exactly one recovering repair, so V2 reduces to label plus evidence (V1 equals V2 on 97.7% of DEV rollouts).
- Verifier v1.1 (accepts the B_DUP alt2 repair that v1 falsely rejects; 0 of 85 verified cheats accepted): prepared, not active. See `evidence/results/VERIFIER_V11.md`.
- A neutral prompt (v2) so the healthy control B_OK is not presupposed faulty; no model has been retrained on it.
- Held-out mechanisms (more fault types) instead of held-out seeds; a larger and independent red team.
- Tier B (multi-turn, real edits) evaluation of strong agents with matched budgets; Tier B RL.
- Repeat the RL comparison with more training seeds and compute-matched controls with multiple runs.
- Independent external review of the sandbox boundary.
