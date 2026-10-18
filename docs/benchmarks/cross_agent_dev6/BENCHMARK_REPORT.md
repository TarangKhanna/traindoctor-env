# TrainDoctor cross-agent smoke benchmark (DEV split, 6 tasks)

**Question.** Can frontier coding agents diagnose and repair a failed distributed-training job (PyTorch/Gloo, CPU), judged by re-running the repaired job rather than by reading their explanation?

**Setup.** Six DEV-split tasks (seeds 10000-10047): two collective-divergence variants, a rank stalled before the collective, a skipped-sample and a duplicated-sample checkpoint/sampler fault, and a healthy control. Each agent gets the same text prompt (job source + config + incident records), one attempt, and returns one JSON object: label, cited evidence ids, claim, unified diff. The diff is applied to a pristine copy (four editable files only) and the job is re-executed in a network-less Modal Sandbox (gVisor). Scoring ladder: V0 label correct; V1 + cited evidence satisfies the scenario's evidence predicate; V2 + verified recovery with sample accounting, checkpoint consistency and bounded lost work. Controls: null (right label, no patch) and gold (oracle). Agents: Claude Sonnet (claude-sonnet-5-5, medium effort, in-session subagent) and Codex (gpt-6.1-sol, medium, via bridge); both on subscription auth.

**Result.**

| agent | V0 label | V1 evidence | V2 verified | recovery only |
|---|---|---|---|---|
| Claude Sonnet | 6/6 | 5/6 | 5/6 | 6/6 |
| Codex | 6/6 | 6/6 | 6/6 | 6/6 |
| gold | 6/6 | 6/6 | 6/6 | 6/6 |
| null | 6/6 | 0/6 | 0/6 | 0/6 |

No invalid, out-of-scope or tampering attempts; no infrastructure failures. The patches of the two agents were identical in 5 of 6 tasks. Claude's single V2 miss (B_DUP) had a repair that recovered but a citation set missing the required replayed-sample records. Codex took 13-22 s per task; Claude roughly 12-24 s (approximate, from harness notifications). Incremental Modal cost $0.06; subscription usage unmetered.

**Reading.** The verifier separates real repairs from non-repairs (null 0/6), but these tasks are at the ceiling: single-site bugs with a short minimal fix. With 6 tasks and heavily overlapping intervals (V2 5/6 vs 6/6) nothing supports ranking the agents. The comparison is controlled on inputs and budgets; isolation is instruction-based for Claude (Read tool not path-restricted) and by construction for Codex (no tools, task inline). DEV scenarios only; nothing is claimed about unseen mechanisms. Discriminating agents needs decoy-laden, multi-site tasks and several attempts per task.

**Reproduce.** See RESULTS.md ("Reproduce"); harness `xbench.py` (paths sanitised to `<repo>`, `<tmp>`), prompts in `prompts/`, answers in `answers/`, scored rows in `results/`, replay outcomes in `outcomes/`, table in `table.md`. Verifier and fixtures: TrainDoctor `modal-infra` branch at b6c1a0d, unmodified. Discarded harness failures (two Codex runs where it could not read files) are described in RESULTS.md and excluded.
