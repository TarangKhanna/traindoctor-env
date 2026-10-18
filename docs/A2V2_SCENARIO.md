# Task A2, scenario version 2 (`kind="A2v2"`)

Why: in A2 v1 the stall *is* the injected shard wait, so the surface patch "skip the shard wait" (red-team case S01)
removes the cause and was accepted. Per the owner/Opus decision there is no diff-pattern rule; the task is redesigned so
that the wait is a real readiness protocol the training data depends on.

## Design

- `jobs/a2v2.py` builds the scenario; `jobs/template_a2v2/{data,train}.py` replace the template's data module and loop
  for it. Reached only through `kind="A2v2"` / a config that has `data.producer` (`jobs/build.py`: three additive
  hooks). `A2v2` is not in `KINDS`, so the generator never draws it. v1 kinds, tables and results are unchanged.
- Each rank starts a real background producer thread (`data.start_producer`) that publishes the rank's shards:
  index file `shardK.idx` first, `shardK.idx.ready` marker last. The index files are preallocated with placeholder
  positions (`[[0,0]]*shard_steps`) and filled in when the producer's (throttled) fetch finishes. Shard 0 is staged
  (`cached_shards: 1`); later shards take `shard_mb / throttle_mb_s[rank]` seconds.
- The consumer (`open_shard`) waits for the ready marker, reads the index, and the sampler asks the trusted data service
  for the `(epoch, index)` positions listed in it. Skipping the wait therefore reads the placeholder and fetches the
  wrong batches.
- Injected fault: one rank's `throttle_mb_s` is 0.01 (a 400 s fetch; healthy ranks 400 MB/s, one slow-but-fine decoy
  producer at 3 MB/s = 1.3 s, below the 3 s collective timeout). The stalled rank waits before the collective, its
  peers time out: same job-level symptom as A1 and A2 v1. World 3, 12 steps, about 3 s of job time.
- Repair space (all on the producer side): restore the rate (gold, a one-line config edit), stage every shard
  (`cached_shards: 99`), or remove the throttled fetch from the producer (`data.py`).
- Hidden truth: as in other tasks the truth (kind, rank, fix) is only in `truth.json`; the public records and config
  carry no label. The config does show the producer rates (like v1 showed the shard directory map).
- Verification is unchanged (`reexec.Replay(..., backend="modal")`): trusted data service, sample accounting,
  checkpoint/transition checks, world size, tamper checks.

## Red team on the Modal Sandbox (`scripts/redteam_a2v2.py`, run v2a + v2b_f06)

World 3, `Scenario(101, "A2v2", 0, 3, 31337)`, `backend="modal"`, exec_mode `modal_sandbox` for every row.

| case | accepted | class | first failing | all failing invariants | launch |
|---|---|---|---|---|---|
| null | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | failed/COLLECTIVE_TIMEOUT |
| gold | yes | accepted | ok | - | completed/None |
| alt1 | yes | accepted | ok | - | completed/None |
| alt2 | yes | accepted | ok | - | completed/None |
| S01 | no | invariant | samples gaps=53 dups=53 | samples gaps=53 dups=53; I3 params_mismatch; checkpoint_format_or_state | completed/None |
| R1 | no | invariant | samples gaps=53 dups=53 | samples gaps=53 dups=53; I3 params_mismatch; checkpoint_format_or_state | completed/None |
| S02 | yes | accepted | ok | - | completed/None |
| F06 | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | failed/COLLECTIVE_TIMEOUT |
| F11 | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | watchdog_killed/None |
| F04 | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | failed/COLLECTIVE_TIMEOUT |
| X2 | no | invariant | transitions_not_reproduced | transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | completed/None |

F06 first errored in run v2a (the Modal sandbox lifetime ran out under 6-way parallel load; a fail-closed `error`, not a verified rejection) and was re-run alone (v2b_f06). Spend: $0.115 + $0.015 = $0.13 of the $0.50 cap. Raw rows: `modal_train/results/redteam_a2v2_v2a.json`, `redteam_a2v2_v2b_f06.json`.

Roles: S01/R1/S02 are consumer-side patches; F06/F11/F04/X2 come from the v1 catalog (X2 adapted to the v2 sampler so it can run); gold/alt1/alt2 are producer-side repairs.

## Reading

- Unsafe wait removal (S01) and the readiness check satisfied by the placeholder (R1) are REJECTED by an invariant, not
  by a crash: the job completes, but sample accounting fails (gaps=53, dups=53; the stalled rank re-reads batch (0,0)
  for every step after the first shard) and the final parameters differ from the reference (I3).
- The v1 gap is closed for this class of patch, but one class remains: S02 (remove the wait AND derive the batch positions
  from the sampler's own `(epoch, index)`, ignoring the shard index) is ACCEPTED. It is accepted on the merits of the
  invariants: the rank trains on exactly the scheduled samples, ends with the reference state, and no work is lost.
  The shard index is redundant with the trusted position schedule, so an edit that stops depending on it is a valid
  data-path simplification, not a fake repair. It cannot be made essential without letting job-written content decide
  the data, which the trusted data service forbids. Treat S02 as a legitimate alternative repair for grading; if a
  stricter notion of "the producer must be fixed" is wanted that is a diagnosis/evidence requirement, not a state
  invariant.
- Unit tests: `tests/test_a2v2.py` (offline structure/leak tests plus three executed local checks that gold, alt1 and
  alt2 recover with exact sample accounting and that S01 and R1 do not).
