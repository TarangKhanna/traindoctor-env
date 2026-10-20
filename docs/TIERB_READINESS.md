# Tier B readiness for multi-turn RL

Scope: `traindoctor_env/tools.py` (`Episode`) over `Replay` in `reexec.py`. Status labels: REPRODUCED (CPU test in
`tests/test_tierb_lifecycle.py` or `tests/test_episode_malformed.py`, failed before the fix where a fix is listed),
INFERRED (read from code, not run), OK (checked, holds).

## Findings

### Fixed (reproduced, failing test first)

1. REPRODUCED: `step` before `reset` raised `AttributeError`; it now returns an error observation.
2. REPRODUCED: a harness failure in `relaunch` or `finish` left an ineffective call in `Episode.calls`,
   double-counting retries. `step` now rolls back the call and re-raises, preserving episode state and
   distinguishing harness failure from agent error.
3. Earlier (80eccec): malformed actions and arguments answered with an error and no state change.

### Open, reproduced

4. No budget: 5000 read calls are accepted without ending the episode (`xfail` test
   `test_episode_is_bounded_in_tool_calls`). `relaunch` and `finish` each re-execute a job; an RL rollout
   requires hard caps, especially on relaunches for cost, to avoid hanging a trainer.

### Open, inferred

5. State cannot be saved or restored. `state()` is a log (calls, actions, observed ids, diagnosis, result). It omits
   `phase`, the per-phase records, and the workspace under `Replay.jd` (a temp dir). There is no `load_state`.
6. A harness exception (sandbox unavailable, bad `resume_from`) propagates out of `step`. The caller cannot tell it from
   a wrong repair unless it catches by type. `finish` with `sandboxed=False` is encoded in `outcome`, but a raise has no
   result row at all.
7. `Replay` (a temp dir) is released only by `reset` or `close`. A finished episode keeps its workspace until then;
   rollout workers that never call `close` leak disk.
8. A relaunch that raises after partly running leaves the workspace changed (new run directory) while `phase` and the
   records are stale. The final verification still sees the changed workspace; the agent's observations do not.
9. `grep_logs` compiles an agent-supplied regex and runs it over every record; a catastrophic pattern blocks the process.
   Requires a pattern length cap or a `re` timeout alternative (for example a linear-time engine).
10. `write_file` has no size cap (`content` is any `str`) and no per-episode write count. Only `job/<name>` is writable
    and symlink writes are rejected by `safefs`; this audit did not re-test those restrictions.
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
`sandboxed=False` or exception -> no reward, rerun.

## Prioritized design for a Tier B benchmark

P0 (before any training use)

- Hard budgets in `Episode`: max tool calls, max relaunches, max `write_file` count and bytes. On exhaustion the next
  step returns an error and the episode finishes through the normal `finish` path (final verification still runs).
- A three-way terminal status on the result: `verified`, `wrong_repair`, `infra_error`. Catch harness exceptions at
  the rollout boundary and write `infra_error`; never score them 0.
- Always `close()` at episode end (context manager or finish hook).

P1

- Deterministic replay as the save/restore mechanism: serialize `(snapshot id, calls)` and rebuild by re-running the
  recorded fix actions. Valid only if `relaunch` is deterministic under the episode salt; this must be tested with two
  independent replays before relying on it. No new storage layer.
- Cap `grep_logs` pattern length and total read output per episode.
- Make the observation after `write_file` say whether record reads are stale (one fixed string, no label content).

P2

- Per-phase `observed_ids` (dict phase -> ids) for audit output.
- Drop or implement `kill_ranks`/`escalate` so every action has an observable effect, or report them as no-ops in the
  tool description.
- Add a mid-episode time budget: relaunch re-execution cost dominates wall clock; report per-step seconds in `state()`.

New sandboxes or runtimes are outside the plan; Modal gVisor remains the only untrusted-execution route.
