# Tier B readiness for multi-turn RL

Scope: `traindoctor_env/tools.py` (`Episode`) over `Replay` in `reexec.py`. Status labels: REPRODUCED (CPU test in
`tests/test_tierb_lifecycle.py` or `tests/test_episode_malformed.py`, failed before the fix where a fix is listed),
INFERRED (read from code, not run), OK (checked, holds).

## Findings

### Fixed (reproduced, failing test first)

1. REPRODUCED: `step` before `reset` raised `AttributeError`; it now returns an error observation.
2. REPRODUCED: a harness failure in `relaunch` or `finish` left an ineffective call in `Episode.calls`,
   double-counting retries. `step` now rolls back the ineffective call and terminates with
   `status="infrastructure_error"`, with no wrong-repair verdict.
3. Earlier (80eccec): malformed actions and arguments answered with an error and no recorded call;
   attempts now consume the call budget.
4. FIXED: `Episode(max_tool_calls=200, max_relaunches=5)` caps all step attempts (including malformed ones)
   and relaunches. The next attempt beyond either cap terminates with `status="budget_exhausted"`
   without further execution or a wrong-repair verdict. The former budget xfail now passes.

### Other findings (fixed or inferred)

5. State cannot be saved or restored. `state()` is a log (calls, actions, observed ids, diagnosis, result). It omits
   `phase`, the per-phase records, and the workspace under `Replay.jd` (a temp dir). There is no `load_state`.
6. FIXED: infrastructure exceptions during relaunch/finish, and unsandboxed finish outcomes, produce terminal
   `status="infrastructure_error"`; `verified_fixed` is null, distinct from `wrong_repair`.
7. `Replay` (a temp dir) is released only by `reset` or `close`. A finished episode keeps its workspace until then;
   rollout workers that never call `close` leak disk.
8. A relaunch that raises after partly running leaves the workspace changed (new run directory) while `phase` and the
   records are stale. The episode now terminates as `infrastructure_error`, preventing further stale observations.
9. FIXED: `grep_logs` uses literal substring matching, with a 256-character pattern cap. Normal text searches
   and rank/window filters are preserved; regex metacharacters are literal, so nested quantifiers cannot execute.
10. FIXED: `write_file` accepts at most 64 KiB of UTF-8 content per call, checked before workspace creation.
    The total call budget also caps write count. Only `job/<name>` is writable; `safefs` rejects symlink writes.
11. `kill_ranks` and `escalate` return ACK with no effect on the workspace. Harmless for scoring, but an agent cannot
    learn from them; they pad the action space.
12. `observed_ids` is a flat set filtered against the current phase only. Ids are phase-keyed so they never collide, but
    the set does not say which phase an id was seen in. `evidence_ok` receives all phases, so this is currently safe.
13. After `write_file` without `relaunch`, record reads still show the previous phase while `job_config` and `read_file`
    show the live file. The module documents this behavior, but the observation does not identify stale records.

### OK

- Schema errors: wrong types, bool-as-int, extra and missing args, non-dict actions, unhashable tool names.
- Finished episode is inert (no state change on any later action). `finish` twice is idempotent.
- `finish` without `submit_diagnosis` yields `no_claim` and is still verified.
- Final reward is computed by re-executing the workspace at `finish`, not from any earlier observation.
- `reset` closes the previous workspace and clears every episode field.
- Edits limited to files directly under `job/`; rejected writes are not recorded as calls.
- Evidence ids differ per phase (`evidence_ids(seed, n, phase)`), so a pre-relaunch id cannot be cited as post-relaunch.

## Execution failure vs wrong repair

`FixOutcome.sandboxed=False` marks an invalid (never verified) result; `ok` is then forced False. A reward function must
not treat that as a wrong repair: it should be dropped or retried. A raised exception is the third state (harness
failure). Required reward mapping: `verified_fixed=True` -> success; `ok=False, sandboxed=True` -> wrong repair;
`sandboxed=False` or exception -> `infrastructure_error`, no reward, rerun. Budget exhaustion has its own
`budget_exhausted` status and no repair verdict.

## Prioritized design for a Tier B benchmark

P0 (before any training use)

- DONE: hard tool-call/relaunch budgets and per-write byte cap. Exhaustion terminates without final re-execution.
- DONE: terminal statuses `verified`, `wrong_repair`, `infrastructure_error`, and `budget_exhausted`.
  Infrastructure failure and budget exhaustion have no repair verdict; never score them as wrong repairs.
- Always `close()` at episode end (context manager or finish hook).

P1

- Deterministic replay as the save/restore mechanism: serialize `(snapshot id, calls)` and rebuild by re-running the
  recorded fix actions. Valid only if `relaunch` is deterministic under the episode salt; this must be tested with two
  independent replays before relying on it. No new storage layer.
- DONE: cap `grep_logs` patterns and use literal matching. Total read output remains a possible follow-up.
- Make the observation after `write_file` say whether record reads are stale (one fixed string, no label content).

P2

- Per-phase `observed_ids` (dict phase -> ids) for audit output.
- Drop or implement `kill_ranks`/`escalate` so every action has an observable effect, or report them as no-ops in the
  tool description.
- Add a mid-episode time budget: relaunch re-execution cost dominates wall clock; report per-step seconds in `state()`.

New sandboxes or runtimes are outside the plan; Modal gVisor remains the only untrusted-execution route.
