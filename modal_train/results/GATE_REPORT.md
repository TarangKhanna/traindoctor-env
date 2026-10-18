# Distractor-menu gate (EXPLORATORY dev_container, not verified recovery)

Status: COMPLETE. GATE RULE RESULT: NO-GO (pooled V1!=V2 2.6% < 10%; old SFT preserving 93.7% > 80%). Task split below, since the decision is Opus's.

## Menu v2 (traindoctor_env/bundle.py, `make_menu(snap, truth, "v2")`; v1 is still the default)
v1's six slots plus three extras shown identically in every variant (so their presence carries no label): `timeout` (pg_timeout_s 3->20, watchdog_s 8->40),
`scratch` (restore() returns 0: train from step 0), `align` (every rank runs the same collective but bucket 0 is broadcast, not averaged). 9 items, shuffled by a
seeded stream; outcomes come from real re-execution (Docker dev container, `td_dev_prepare` with an untracked `modal_train/prepare_opts.json`, removed afterwards).
Tests: tests/test_menu_v2.py (edits apply exactly once, valid Python, equal item length, extras identical across variants, statistical menu-leak audit on 240 seeds,
all variants fit 3,072 real Qwen3 tokens: max A 2093, B 2402 on synthetic bundles).

## DEV tables (modal_train/cache_menu2, 48 episodes, 432 candidates, 57 recovering, 10 fault_not_injected)
Candidates whose re-executed job completed (final checkpoint committed) and their recovery verdict, per kind (episodes):
| kind (n eps) | timeout completes | scratch completes | align completes | noop completes | exclude completes | any recovering distractor |
|---|---|---|---|---|---|---|
| A1 (10) | 0 | 0 | 3 | 0 | 2 | 0 |
| A2 (10) | 0 | 0 | 0 | 0 | 3 | 0 |
| B_SKIP (10) | 10 | 10 | 10 | 10 | 6 | 0 |
| B_DUP (9) | 9 | 9 | 9 | 9 | 6 | 0 |
| B_OK (9) | 9 | 9 | 9 | (is correct) | 6 | 9 (timeout raise is a harmless change to a healthy job, so it recovers) |
Findings: (1) Task B: every episode has 4-5 completing but invalid repairs (the wanted design). (2) Task A1-cfg and A2: the only completing wrong repair is the
destructive rank exclusion for some; "raise timeout" and "restart from scratch" hang (no completed relaunch). This is structural: A2's shard files are cosmetic, so
every edit that unblocks the stall is a valid repair; no completing-but-invalid A2 distractor was found. (3) B_OK now has two recovering repairs (relaunch and timeout raise).
Old-menu baseline recomputed offline from saved evals: esft_full V1!=V2 1.3%, preserving-given-label-correct 89.9%; eg_v2 2.3% / 88.8%.

## Results (9 evals of OLD-menu-trained checkpoints on menu v2 + prompt v2, 48 DEV eps x G=8 = 384 rollouts each; L40S, ~280 s, $0.13-0.19 each)
Intervals are 95% percentile bootstrap resampling EPISODES. Raw: GATE_REPORT_policies.json (per policy), GATE_POOLED.json (pooled/split), gate_*_eval.json.
Pooled = 8 trained policies (esft_full, csft, g_v0 x3, g_v2 x3), 3072 rollouts.

| slice | V1!=V2 (all rollouts) | V1!=V2 among V1-true | invariant-preserving given label-correct |
|---|---|---|---|
| pooled | 2.6% [0.6, 5.3] | 18.9% (n_v1=424) [4.9, 35.9] | 87.3% (n=993) [76.4, 95.3] |
| Task A (A1+A2) | 6.3% [1.8, 12.0] | 39.0% (n_v1=205) [13.8, 64.1] | 68.6% (n=398) [49.3, 86.8] |
| Task B faulted (B_SKIP+B_DUP) | 0.0% | 0.0% (n_v1=37) | 99.5% (n=183) |
| B_OK (no fault) | 0.0% | 0.0% (n_v1=182) | 100% (n=412; label-correct = chose a recovering repair, 2 exist) |
| old SFT (esft_full) overall | 1.3% [0.3, 2.9] | 12.2% | 93.7% (n=95) |
| esft_full Task A | 3.1% [0.6, 6.3] | 38.5% | 72.7% (n=22) |
| csft overall / Task A | 2.1% / 5.0% | 19.0% / 61.5% | 89.0% / 66.7% |
| V0 arm (3 seeds) / V2 arm (3 seeds), pooled | 2.5% [0.6, 5.0] / 3.3% [0.7, 7.0] | 17.7% / 21.5% | 86.6% / 86.1% |
base (no adapter): valid 20%, V0 7%, V1 1%, V2 0%, V1!=V2 0.5%.

Per-policy rates (V0/V1/V2): esft_full .25/.11/.09, csft .26/.11/.09, g_v0 .39/.15/.14 (s1 .36/.15/.12, s2 .33/.12/.09), g_v2 .34/.16/.13 (s1 .34/.16/.12, s2 .31/.14/.11).
Role of the chosen repair among label-correct rollouts (pooled): almost all `correct`; the rest split between decoy_* (decoy_sync, decoy_perrank, decoy_ckpt) and distractors (align most often, then timeout, scratch); roles per policy in GATE_REPORT_policies.json (`label_correct_chose_role`).
B_OK is its own column/row above: it has two recovering repairs (relaunch, timeout raise), so it never produces V1!=V2.
Where V1!=V2 occurs it is essentially all Task A1 (A1: 4-16% per policy, 12-16% for g_v0_s1/s2 and the three g_v2 seeds): the label+evidence is right but the chosen repair does not recover (preserving 17-57% in A1). Task B contributes ~0 disagreement: every Task B faulted rollout that is label-correct also picks the correct repair (99.5%), because the distractors there complete but the models rarely choose them.
Caveat (structural, found by previous agent): Task A menus lack >= 2 completing repairs; A2 has no completing-invalid distractor.

## GO/NO-GO by the rule: NO-GO
(a) pooled V1!=V2 = 2.6% (< 10%) -> fail. (b) SFT preserving = 93.7% (> 80%) -> fail. The distractor menu does not create a V1/V2 gap on these old-menu-trained checkpoints. Task A alone gives 6.3% (still < 10%, CI to 12%) and 68.6% preserving (would pass (b)); Task B alone is 0%. These are checkpoints trained on the old menu, so this measures menu difficulty vs. existing policies only. EXPLORATORY dev_container tables, not verified recovery.

## Live Modal Sandbox audit (dev_container table cell vs backend='modal' re-execution)
modal_train/cpu_app.py::run_audit (episodes built and replayed inside one Modal CPU function; the replays run in network-less gVisor Sandboxes, exec_mode=modal_sandbox, sandboxed=True). 16 cells, 1 episode per kind (A1 10001, A2 10006, B_SKIP 10000, B_DUP 10008, B_OK 10002): every recovering cell (correct x5, B_OK timeout raise) plus 2 non-recovering per episode (decoy_ckpt, exclude_rank, decoy_perrank, distractor_align, decoy_sync, distractor_timeout, distractor_scratch).
Result: 16/16 agree on `ok`, and all of ok, steps_completed, ckpt_valid, lost_steps, ranks_consistent, samples/params match, fault_not_injected are identical. 0 disagreements, 0 errors. Not covered: A-kind recovering distractors (none exist), non-recovering-but-completing Task B cells beyond one distractor_timeout/align each (the other completing-invalid kinds - scratch for B_SKIP/B_DUP - were not sampled). Raw: audit_live_menu2.json.

## Spend (ledger actual rows, this task)
9 evals $1.389 (base .190, esft_full .154, csft .149, g_v0 .155/.153/.133, g_v2 .149/.153/.153) + audit $0.154 (booked estimate: ~600 s container + 387 s sandbox time at 4 CPU, $0.14/CPU-h as the ledger models it) = ~$1.54 of the $1.60 task cap; ledger total $9.58 of $30.
