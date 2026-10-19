# Status-only rule (post-hoc diagnostic audit, predefined before evaluation)

Post-hoc diagnostic audit, not part of the original experiment. Evaluated on new development seeds
(80000-80999), not the consumed held-out set.

Question: does process/job status alone identify the fault kind {A1_src, A1_cfg, A2}?

Input: the agent-visible Tier A bundle text only (`make_bundle(...).text`). Nothing else: no scenario, truth,
label, kind name or seed. Record lines in the `== records ==` section look like `<id> <text>`.

## Classifier 1: process-status-only
Looks only at per-rank process lines, regex `\[run\d+\] rank \d+ (failed|killed) exit=(-?\d+)`.
1. If any rank line says `killed`, or has a negative exit code (signal / watchdog kill): predict A2.
2. Else (every rank failed with a positive exit code, e.g. exit=1 COLLECTIVE_TIMEOUT): predict A1_src.
   Status gives no way to separate A1_src from A1_cfg, so the A1 default is fixed in advance to A1_src.
3. No process lines at all: predict A1_src.

## Classifier 2: job-summary-only
Sees only the job-level record (`[runN] resume_from=... -> failed ... error=...`) and the scheduler lines
(`submitted:` / `failed:`). No per-rank process lines.
1. If any of those lines contains `killed`, `stalled`, `watchdog` or `exit=-`: predict A2.
2. Else predict A1_src.

Both rules are final. No tuning after evaluation; if wrong, report as is.
