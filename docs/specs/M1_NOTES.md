# M1 v2 implementation notes (deviations, semantics, measurements)

## Layout
- `jobs/template/` is the job (`train.py`, `data.py`, `ckpt.py`, `config.yaml`, plus `config.py`, a tiny YAML-subset
  reader so no PyYAML dependency is needed). `jobs/build.py` turns a seed into `config + Edit list`; the bug is a
  textual edit on the pristine files and the repair is the same edit reversed. `jobs/m2_candidates/` holds the old faults.
- Harness-owned (outside the workspace, never editable): `jobs/launcher.py` (subprocess ranks, trusted data server loop,
  watchdog, step-boundary rendezvous), `jobs/dataserver.py` (the data server), `jobs/trusted.py` (ledger + checkpoint
  capture), `jobs/sandbox.py`, `jobs/resubmit.py` (lock, manifest validation, automatic restart),
  `jobs/runtime/` (the only repo code a job process may import: `td_client.py`, `harness.py`),
  `traindoctor_env/reexec.py` (verification), the reference run and `fixture/_private/`.
- `EDITABLE = job/train.py, job/data.py, job/ckpt.py, job/config.yaml` (`jobs/build.py`). `job/config.py` is trusted.

## Kinds and labels
A1 (source branch / per-rank config mode), A2 (stale shard dir), B_SKIP, B_DUP, B_OK. Labels: COLLECTIVE_DIVERGENCE,
RANK_STALLED_BEFORE_COLLECTIVE, CKPT_SAMPLER_INCONSISTENT (skip and dup), NO_FAULT. Scenario kind, world size and job
seed come from separate private streams (`scenario:`, `job-seed:`, `build:` prefixed `random.Random`); the public job
seed is not the generator seed.

## Trust boundary (integrity-hardening pass)
Nothing that runs inside a job process is trusted. Three launcher-side sources decide verdicts, all outside the sandbox
the job runs in:
1. **The data server** (`jobs/dataserver.py`, served from the launcher's own loop over one private `socketpair` per rank,
   so the asking rank is the socket, never the message). Samples are generated from `(job seed, per-scenario salt, sample
   id)`; ranks request only a position `(epoch, index)` of their own shard. The salt is `sha256(key | scenario seed)`;
   the key comes from `TD_KEY` or `~/.traindoctor/key` (0600) and is never placed in a job's environment, arguments or
   files, and `~/.traindoctor` is unreadable under the sandbox. It is per scenario, not per execution, on purpose: the
   reference run, the incident and every verification run must see the same data, and the public snapshot must be
   byte-identical for a seed. Requests are validated (exact fields, integer in-range positions, <= 4096 bytes,
   <= 10,000 per rank per run); malformed ones get an error, oversized ones cut the connection (the rank fails with
   PROTOCOL_VIOLATION). Replayed and out-of-order positions are legal (a position is job state; B_SKIP/B_DUP *are*
   wrong positions) and are logged. Tests: `tests/test_dataserver.py` (each probe states why it is rejected or harmless).
2. **Checkpoint capture** (`jobs/trusted.py`): while rank 0 is parked in its data request (its checkpoint is written, the
   barrier passed) and once more after every process of the run is dead, the launcher copies each checkpoint the manifest
   lists, byte for byte, hashed, into `<job_dir>.trusted/captures/`. The ledger (`ledger.jsonl`, same sibling dir) records
   for every run: launch parameters, how it ended, the data server's log, the captures. The sibling dir is not under the
   job dir, so a job cannot write it and, sandboxed, cannot read it.
3. **Process history** (the ledger again): lost work and relaunch counts come from it, never from `runs/` files the job can
   write. `runs/<id>/{job,start}.json`, `stderr/`, `samples/` are still written (for the agent / public records) but
   nothing reads them back for a verdict.

The in-process `jobs/runtime/harness.py` only sends a step counter and a state digest after each optimizer step. That is
evidence for the agent-readable records (`last_step`, the `samples` records) and for `ranks_consistent`; a job can silence
or forge it, so no verdict depends on it except that forged disagreement can only fail the job.
Verification (`Replay.outcome`, used by `Episode.finish`, `verify_fix`, the Tier A table) runs in the launcher's process:
it copies the pristine incident (never `_private/`; `trusted.copy_job`), applies only the agent's EDITABLE files,
re-executes and judges from the ledger. Non-editable changed files are discarded and listed in `FixOutcome.tampered`; the
old textual tripwire (grep for `.harness`) is gone.

Process isolation (`jobs/sandbox.py`, `sandbox=True` for every agent-edited run): macOS `sandbox-exec`, DENY BY DEFAULT;
allows only exec/fork of the interpreter, reads of the run dir, `jobs/runtime/` and the Python install (+ the system
paths dyld/libc need), writes to the run dir, loopback networking and a short mach-lookup list. FAIL CLOSED: where it
cannot be applied `SandboxUnavailable` is raised; `allow_unsandboxed=True` (or `TD_CI_UNSANDBOXED=1`, set by CI for
repo-owned code) is the explicit opt-out. The child environment is an allowlist (`PATH`, rendezvous variables,
`TD_FD`, thread settings), never `os.environ`. **Status: the profile could not be exercised while it was written**
(`sandbox-exec` cannot nest inside another sandbox, and the authoring session was itself sandboxed); the behavioural
probes in `tests/test_sandbox.py` (read `_private/` or a sibling, write outside the run dir, non-loopback socket,
`/usr/bin/open`, `osascript`, any exec but the interpreter) run wherever the sandbox can be applied and are skipped
otherwise. Expect to tune `MACH_LOOKUPS` / `SYSTEM_READS` the first time a job runs under it
(`log stream --predicate 'sender == "Sandbox"'`).

## Sampler, checkpoint and grading semantics (what the tests check against)
- Sampler: rank r owns the ids with `id % world_size == r`, shuffled per epoch with a generator seeded by
  `(seed, epoch, rank)` (now `jobs/dataserver.schedule_ids`); position is `(epoch, index)`; an epoch is
  `num_samples // world // batch_size` steps. The job's `data.Sampler` is only the position; `next_batch()` asks the server.
- A checkpoint commits, as of the end of step `s-1` (file `ckpt/ck<s>/state.pt`, manifest entry written by rank 0 after the
  file): step `s`, model, optimizer momentum, sampler position `(epoch, index)`.
- **Reference**: the pristine job, uninterrupted, checkpointed at EVERY step by the launcher (`ckpt_every=1` does not change
  training); `reference.json` holds the full committed state after every step plus the sample schedule.
- **Fast path (sufficient)**: the run completed, the final checkpoint (step == steps) is committed, and EVERY committed
  checkpoint of the verification run -- model, momentum, sampler position, same keys -- equals the reference state at its
  step within `FAST_TOL` (= `TOLERANCE` = 1e-5). Bit-identity is reported separately (`bit_identical`).
  Why this suffices: the job is deterministic (the per-step data is a pure function of `(seed, epoch, rank, index)` served
  by the data server; there is no other RNG), so the state at commit k+1 is a function of the state held or restored at
  commit k and the samples consumed in between. Equal state at every commit therefore means the right samples were
  consumed in the right order since the previous commit, and the sampler position was saved and restored correctly.
  Uncommitted work (after the last checkpoint before a failure) is lost by definition and recomputed by the restart.
  B_SKIP/B_DUP and every tamper probe are caught at the first commit that follows the bug.
- **Fallback (a valid repair need not be state-equal, e.g. a different checkpoint layout or a numerically different but
  equivalent update)**: pass by the independent invariants, all of which must hold: (a) every committed checkpoint carries
  model/optimizer/sampler entries (extras allowed), finite values, `step` equal to the commit step, and a sampler position
  exactly where `step` batches leave a sampler; (b) every transition between consecutive committed states (from the
  checkpoint a run resumed from, or the initial model, to its first commit, then between commits) is reproduced by the
  PRISTINE update rule (SGD, momentum 0.9, gradients averaged over ranks, lr/hidden/seed from the reference config) applied
  to the batches the data server logged, within 1e-5 -- restored-state consistency at each resume; (c) logical sample
  accounting: the batches the data server served, in order and cut at the committed-step boundary (a completed run commits
  everything, a failed run what it had checkpointed; each run supersedes its predecessor from its start step), equal the
  reference schedule exactly (`sample_gaps == sample_dups == 0`). `tests/test_tamper.py` exercises this path with a repair
  that adds an entry to the checkpoint.
- Both paths additionally need: lost work <= `ckpt_every` summed over every recovery attempt; all ranks completed and
  reported the same final digest; no tampering. Prefetch (served != consumed) cannot fail the fast path; on the fallback
  path a job that interleaves unused requests defeats the in-order accounting (the documented limit of that path).
  `FixOutcome.path` says which path accepted the run (`state_equal` | `invariants`).
- Lost work per run = steps in `[resume, min(end, furthest progress so far))`, `end = start + max served - (1 if the run did
  not complete: the request in flight was not a step)`. Task A: every relaunch since the incident, no exemption. Task B:
  the same, over the agent's workspace relaunches and the protocol run, except `B_VALIDATION_SCRATCH_RUNS = 1` from-scratch
  rerun (the repair legitimately needs one: checkpoints the buggy code wrote are untrustworthy); progress made by that run
  restarts the frontier, so its own automatic restart is charged normally. A second scratch rerun costs a full run.
  A job that recomputes internally without asking the server again is not seen by this metric (documented gap).

## Deviations from the spec
- Idempotency keys and the keyless "resume from the previous run's start" resubmit semantics were dropped with the old
  faults; `relaunch(resume_from=None)` resumes from the latest valid checkpoint, `"none"` starts from step 0.
  Must-fix 3 is restated as "repeating a completed relaunch keeps the recovery".
- Must-fix 4 (lock liveness): the owner holds an `flock` on `.lock`; a lock nobody holds is stale whatever pid it records.
- Lost work (must-fix 2) is summed over every relaunch for Task A and Task B alike; Task B exempts exactly one
  from-scratch validation rerun (see "grading semantics"). This supersedes the earlier "only the protocol's restart counts".
- `pg_timeout_s=3`, `watchdog_s=8` (no-progress window; 30 s startup grace). Collectives use a `new_group` with that
  timeout while `init_process_group` waits 60 s, so slow-starting ranks under load do not time out at init.
- Worker kill (Task B) is a deterministic step-boundary rendezvous, not a timer: every rank is parked when it asks for the
  batch after step K (so each checkpoint written so far is complete and nothing is running), the launcher captures
  checkpoints, kills the victim, waits for it to be dead, then releases its peers. If a rank is gone before the boundary the
  plan is abandoned and the verifier reports "the incident's worker failure did not happen". `progress/rank<r>.json` is
  written after the batch is fetched, so at the boundary it holds the position after step K (the B_SKIP bug is
  deterministic). Nothing in the failure path depends on sleeps.
- Evidence predicates only count records of the incident's own runs (what the agent's later runs write never counts).
- Citation cap: the deduplicated list of cited record ids may hold at most `EVIDENCE_CAP = 8` (public; stated in the task
  text and the bundle header) at the shared evidence gate, so V1 and V2 both reject cite-all-ids. Every oracle answer
  cites 2-4 ids (`tests/test_tiera.py::test_cite_all_exceeds_the_public_cap_...`).
- Hang detection depends on Gloo timing out on mismatched collectives; measured reliable for these shapes
  (broadcast on bucket 0, reduce on the last bucket). Under extreme machine load (load average > 20) runs can be
  killed by the watchdog instead; tests then fail loudly rather than pass.

## Tier A bundle and menu (`traindoctor_env/bundle.py`)
- Menu: exactly `K_MENU = 5` candidates for EVERY variant (the menu size said which variant it was). Task A: correct,
  no-change relaunch, raise the timeout, fix the other cause, exclude a rank [destructive]. Task B: correct, change lr,
  plus {no-change, raise the timeout, `correct_alt` [B_SKIP: stop writing the progress file the restore reads]},
  {no-change, raise the timeout, exclude a rank} [B_DUP], or {change batch size, change model width, exclude a rank} [B_OK, whose correct answer is the
  no-change relaunch and whose distractors all alter training; raising the timeout would be a second correct answer there]. Every variant therefore shows exactly one no-change item. Ids `R<5 hex>` from (seed, role), order from
  `Random("menu:<seed>")`. Outcomes come from real re-execution (`outcome_table`, two jobs at a time);
  `answer_verdict(..., live=snapshot)` re-executes instead of using the table.
- Metadata hardening: every menu item is rendered with exactly `ITEM_CHARS = 640` characters (the changed lines, real unchanged
  lines around them while they fit, then blank padding), `job/config.yaml` is shown with each line padded to `CONFIG_WIDTH`, and the bundle holds `QUOTA[kind]`
  records of each of log/trace/samples (at least the certificate's) so record counts do not separate A1 from A2. The
  metadata-only predictor test (`tests/test_bundle_leak.py`, sweep: 60 executed scenarios, leave-one-out best-threshold
  stump per feature and 1-NN over all features, per task family) must not beat the majority class by more than 0.15.
- Bundle text (`make_bundle`): header + labels, `job/config.yaml`, records with ids, the menu diffs. Full source does
  not fit the cap; the diffs carry the relevant code. Cap 3,072 tokens (raised from 2,048, docs/DESIGN_NOTES.md) by `est_tokens` (the Qwen3 tokenizer is not
  available offline in this venv): 4 chars/token for prose plus 4 tokens per evidence id (hex ids tokenize badly).
  This is an estimate; re-measure with the real tokenizer in M3.
- Selection: always kept = incident summary (job/process/scheduler/ckpt records) + the evidence certificate (every record of
  any alternative of the mechanism's predicate). Then, per kind, other records are drawn (stream `Random("bundle:<seed>:<kind>")`,
  independent of the label) up to QUOTA; retained records are shown in canonical order. If the result exceeds the cap the
  build raises `BundleTooLarge`.
- Sweep numbers (`pytest -m sweep -s`) must be re-measured after this change (menu items are longer, record fill is
  smaller); the figures recorded before it (1181-2048 est. tokens) no longer apply.

## Claimed vs verified
`submit_diagnosis` takes `claimed_fixed` (the agent claims the final job recovers correctly; for B_OK that is True
with no edits). `finish` runs the final verification and stores `result.claimed_vs_verified` in
`Episode.state()`: match | overclaim | underclaim. Tier A `answer_verdict` takes `claimed_fixed` and returns the same field.

## Known gaps after the integrity pass
- The sandbox profile is untested against a real job (see "Trust boundary"); until it has run once outside another sandbox,
  agent-edited code must not be run on the owner's machine. Verification fails closed where the sandbox cannot be applied.
- The salt's key lives in `~/.traindoctor/key` (or `TD_KEY`). An env-var key is visible to `ps eww` for same-user processes
  on macOS; the file is the supported route. A job that is NOT sandboxed (CI, own code) can read the file.
- Lost work is measured in data-server requests. A job that recomputes steps internally without asking for data again (a
  private cache of served batches) hides that recomputation; wall-clock is not measured.
- On the fallback path, sample accounting cuts the served log in order at the committed-step boundary, so a job that
  interleaves unused requests is only accepted through the fast path.
- Checkpoint capture happens at rank 0's next data request and at process exit. A job that rewrites a checkpoint file
  between its commit and that point is graded on what was captured; whatever it restarts from is what the next commit
  reflects, so a mismatch surfaces one commit later.
- Gloo ranks bind loopback ports with no per-port restriction in the profile (ephemeral ports are not predictable).
