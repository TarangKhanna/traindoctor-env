# Status-only audit, run1

Post-hoc diagnostic audit, not part of the original experiment, run on new development seeds (80000-80999),
not the consumed held-out set. Rule (committed before evaluation): `audit/status_only_rule.md`, rule commit
e673dbfb385460afa98992ea310fe1cf4543f36b. Inputs: bundle text only. 40 seeds per class (A1_src = A1 variant 0,
A1_cfg = A1 variant 1, A2); seed list in `seeds.json`. Raw: `raw.jsonl`; reproduce: `python3 audit/status_only_recount.py results/status_audit/run1`.

n=120
### pred_process
rows = truth, cols = predicted
| truth | A1_src | A1_cfg | A2 | acc |
|---|---|---|---|---|
| A1_src | 40 | 0 | 0 | 40/40 = 1.000 |
| A1_cfg | 40 | 0 | 0 | 0/40 = 0.000 |
| A2 | 0 | 0 | 40 | 40/40 = 1.000 |
overall: 80/120 = 0.667

### pred_job
rows = truth, cols = predicted
| truth | A1_src | A1_cfg | A2 | acc |
|---|---|---|---|---|
| A1_src | 40 | 0 | 0 | 40/40 = 1.000 |
| A1_cfg | 40 | 0 | 0 | 0/40 = 0.000 |
| A2 | 40 | 0 | 0 | 0/40 = 0.000 |
overall: 40/120 = 0.333


## What it means
Process status alone separates A2 from A1 perfectly on these 120 seeds (120 of 120 correct as a binary A1-versus-A2 call: 80 A1 and 40 A2 seeds; the three-class result is 80 of 120), because in
every A2 bundle exactly one rank is `killed exit=-9` (watchdog kill, no error class) and no A1 bundle has a killed
rank. It cannot separate A1_src from A1_cfg (all ranks `failed exit=1 COLLECTIVE_TIMEOUT` in both; the rule's fixed
A1_src default gets A1_src right and A1_cfg wrong), so 3-class accuracy is 0.667 = (A2 + A1 collapsed) ceiling.
The job-level summary and scheduler lines are identical across kinds (`error=COLLECTIVE_TIMEOUT`), so that
classifier is at the 0.333 default. Environment/bundle unchanged. The bundle therefore leaks the A1-vs-A2 kind
through per-rank process status (not through job summary).

## A2 timeout vs killed (from the real bundles)
In all 40 A2 bundles: 1 rank is killed, the other 2-3 ranks time out. The killed rank is the stalled one
(seed 80902: truth stalled rank 0; its bundle also shows `waiting for shard 1 ...`).
Seed 80902 (world 3):
    [run0] rank 0 killed exit=-9 last_step=6
    [run0] rank 1 failed exit=1 last_step=6 COLLECTIVE_TIMEOUT
    [run0] rank 2 failed exit=1 last_step=6 COLLECTIVE_TIMEOUT
Seed 80900 style A1 for contrast: every rank `failed exit=1 last_step=4 COLLECTIVE_TIMEOUT`.
So README wording: in A2 the healthy peers hit COLLECTIVE_TIMEOUT and the stalled rank is killed by the watchdog
(exit=-9); in A1 all ranks time out.
