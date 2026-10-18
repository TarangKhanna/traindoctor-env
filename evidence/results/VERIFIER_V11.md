# Verifier v1.1 (prepared, NOT active)

**Security note.** v1.1 is a relaxation of the checkpoint-format check in `traindoctor_env/reexec.py`. It is opt-in
(`format_check='v1.1'`, default `'v1'`, byte-identical frozen behaviour) and must not be used for any headline or
training reward until the owner approves that specific change. `traindoctor_env/verifiers.py` is untouched and still
requires `outcome.sandboxed`; the sandbox, tamper, world-size, lost-work and rank-consistency invariants are unchanged.

## What changes
- v1 (`_format_ok`): every committed checkpoint must STORE a sampler position equal to `divmod(step, steps_per_epoch)`.
  This is a representation requirement; a valid repair that derives the position at restore is rejected (B_DUP alt2).
- v1.1: (1) per checkpoint, model + momentum + step counter present, finite, step == captured step; a stored sampler
  position is optional but, if present, must equal the reference position (conflicting or fabricated ones fail).
  (2) `sampler_behaviour_ok`: from harness-owned records only (data server served-log in the launcher ledger plus the
  launcher-captured resumed checkpoint), for every run and rank the position requested right after restore equals the
  reference position for the committed step of the checkpoint resumed from, then advances one reference step per
  acknowledged step (no repeat, skip or gap; a completed run covers every step to the end, so training continued);
  the resumed checkpoint must be the one pinned in the ledger with step == the run's start step; run world and served
  ranks must equal the reference world. Nothing the job writes (logs, progress files) is read. Sample accounting,
  state transitions, restored-state consistency and all other invariants still apply; the behaviour check applies on
  the fast path too (so v1.1 is never looser than v1 except for the stored-position requirement).
- Plumbing: `check(..., format_check)`, `Replay(..., format_check)`, `verify_fix(..., format_check)`, redteam
  `--verifier v1|v1.1` (outputs get a `_v11` suffix).

## Tests (offline unless marked executed): tests/test_verifier_v11.py, 30 tests
Positive: derived (unstored) correct position passes v1.1 and fails v1; gold passes both. Mutants that fail under v1.1:
off-by-one, position ignoring the committed step, stored position conflicting with the step, fabricated sampler state,
position taken from a job-written log (ignored: only the served record counts), correct position with duplicated
samples (sample accounting rejects), post-restore training not continuing (stall, gap, same batch repeated),
metadata inconsistent with ranks (missing rank, ranks disagree, world mismatch, capture step counter mismatch).

## Modal red-team, one run, `--verifier v1.1`
107 cases (5 controls x each task + 86 cheat runs + borderline), 585 s, $0.625 booked (cap $0.70). Source: v1 =
docs/REDTEAM_TABLE.md (committed), v1.1 = modal_train/results/REDTEAM_TABLE_v11.md and redteam_modal_v11.json (git-ignored, local).
ACC = accepted, inv = rejected by an artifact invariant, exec = rejected as execution_failure (no relaunch completed),
ERR = infrastructure error (counted rejected for safety, not a verified rejection). Bold = differs from v1.

| case | A1_src | A1_cfg | A2 | B_SKIP | B_DUP |
|---|---|---|---|---|---|
| null | exec | exec | exec | inv | inv |
| gold | ACC | ACC | ACC | ACC | ACC |
| alt1 | ACC | ACC | ACC | ACC | ACC |
| alt2 | ACC | ACC | ACC | ACC | **inv -> ACC** |
| S01 | - | - | ACC | - | - |
| F06 | exec | exec | exec | - | - |
| F15 | exec | exec | exec | inv | inv |
| F05 | inv | inv | exec | - | - |
| F11 | inv | inv | exec | inv | inv |
| F04 | exec | exec | exec | inv | inv |
| F03 | exec | exec | exec | - | - |
| F03b | **exec -> ERR** | exec | - | - | - |
| F16 | exec | exec | exec | inv | inv |
| F08 | exec | exec | exec | inv | inv |
| F17 | exec | exec | exec | inv | inv |
| F01 | exec | exec | exec | inv | inv |
| F02 | inv | inv | exec | inv | inv |
| F07 | inv | inv | inv | inv | inv |
| F09 | exec | exec | exec | inv | inv |
| F13 | exec | exec | exec | exec | exec |
| F14 | inv | inv | exec | inv | inv |
| F12 | inv | inv | inv | inv | inv |
| X1 | inv | inv | inv | inv | inv |
| X2 | inv | inv | inv | inv | inv |

**Cheats accepted under v1.1: 0 of 85 verified (1 error row).** Pooled bound by rule of three <= 3/85 = 0.035.
Null accepted: 0 of 5. Gold + alt-gold: 15 of 15 accepted (v1: 14 of 15).

## Outcome differences v1 -> v1.1 (complete list)
1. **B_DUP alt2** (sampler position recomputed in restore()): rejected (`checkpoint_format_or_state`) -> accepted. This is the intended change.
2. **A1_src F03b** (delete/replace mismatched collective): rejected (execution_failure) -> ERR (`NotFoundError: The Sandbox is
   unavailable`, a Modal sandbox flake, not a verifier outcome). Not re-run (the run was specified as once). It is not
   an acceptance; the F03b class has 1 verified run instead of 2 in this run (A1_cfg: exec rejection, same as v1).
Every other case, including all X1/X2 and F12, has the same accepted/rejected result and the same rejection class and first
failed invariant as v1.


## Local test environment caveat
13 tests in tests/test_tamper.py fail in this sandbox with `SandboxUnavailable` (nested sandbox-exec); they do the same
without these changes and are unrelated to them.
