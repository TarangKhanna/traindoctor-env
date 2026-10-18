# TrainDoctor verifier audit: what was checked, what was caught, what is not verified

Scope: the Tier A prompt bundle and menu (can a model read the answer off the menu alone?), the V0/V1/V2 verifier and tamper boundary, and the audit tests themselves. All numbers are read from files in the repo; paths are given per claim. `mi/` below means the worktree the repo root. This is an audit by the project's own authors and agents plus Codex boundary reviews; no external red-team has run yet (the design spec schedules one after M2).

Three verifiers (the design spec (docs/DESIGN_NOTES.md)): V0 = label correct; V1 = V0 + cited evidence observed and relevant; V2 = V1 + verified repair by re-execution with recovery-correctness invariants (committed sample accounting, restored state vs reference, lost work <= checkpoint interval, rank consistency, tamper checks) + safety gates.

## 1. Menu-leak validation (can the menu alone predict the answer?)

Setup: 400 DEV seeds (10000..10399), 5-fold cross-validation across seeds, Wilson 95% CIs, CPU only. Majority class = 0.360 for repair slot and label. Files: `mi/experiments/verifier_audit/leak_validation_796ba23.md` (failure record, kept unchanged), `leak_validation_97a49d8.md` (re-validation), with `.json` and `.py` alongside.

Round 1 (code 796ba23): FAIL, one cause. Boosted stumps on menu-only features reached 0.495 [0.446, 0.544] on repair slot and 0.485 on label, versus majority 0.360. Cause isolated by ablation: the no-change and exclude menu items drew context lines from the buggy job text while the location items used clean text, so `reduce_bcast` appeared in those two items in 50/50 A1-per-rank seeds with 0 false positives. Deleting the two items dropped stumps to 0.350. A second finding: candidate ids were a public hash of (seed, role), recovering the seed and the correct id in 100/100 menus.

Fix (code 97a49d8, `traindoctor_env/bundle.py`): all menu item contexts rendered from the clean text in one neutral form; candidate ids keyed with a TD_KEY-derived per-scenario secret.

Round 2 (code 97a49d8), menu-only out-of-fold accuracy (`leak_validation_97a49d8.md`):

| target | majority | boosted stumps | NB | LR |
|---|---|---|---|---|
| repair slot (pre-fix stumps 0.495) | 0.360 | 0.333 [0.288, 0.380] | 0.237 | 0.273 |
| label (pre-fix stumps 0.485) | 0.360 | 0.300 [0.257, 0.347] | 0.258 | 0.223 |

Counterfactual groups (200 groups x 6 variants, chance 0.333): repair stumps 0.273, label stumps 0.266, all at or below chance. The old public id formula recovers the seed from 0/100 menus (was 100/100). Before/after: `mi/experiments/verifier_audit/menu_leak_before_after.json` (n-gram LR above-chance margin -0.06 after the fix; the pre-fix content feature `added_text` reached 0.74 vs chance 0.325, post-fix 0.175).

Known and kept (not a menu leak): the bundle config alone reveals A1-cfg (`reduce_bcast`) and A2 (`part-<world>`). Menu + config stumps reach about 0.71 (`config_visibility_note.md`, `leak_validation_97a49d8.md`). This is real causal evidence the owner decided to keep visible. The preregistration (`mi/evidence/prereg/strata_v0_v2.md`) therefore reports V0-vs-V2 overall and on two strata: config-obvious (A1_cfg 50 + A2 86 = 136 DEV episodes) versus evidence-ambiguous (A1_src 35, B_SKIP 63, B_DUP 81, B_OK 85 = 264). A config-only rule gets label accuracy 1.000 on config-obvious TRAIN scenarios (n=124) and 0.533 on evidence-ambiguous (n=276), 0.678 overall (`mi/evidence/prereg/strata_v0_v2.md` section 2; `strata_config_baseline.json`).

## 2. Mutation tests of the audit tests (17 mutants)

Question: would the leak and config-visibility tests notice if the bug came back? `mi/scripts/mutation_check_leak.py` applies 17 in-memory mutants (library code on disk is never changed) and requires each test to pass on the real code and fail on the mutant. Output `mi/experiments/verifier_audit/mutation_check_leak.json`: 12 tests pass on real code; 17 mutants, all detected, `undetected: []`. Covered mutants include a label-encoding config key (CS1), a label name leaked into a config value (CS3), a config cause that follows the label (CS5), a repair that does not remove the config cause (CS6), buggy-text context for the no-change and exclude items (ML1, ML2 = the 796ba23 bug), unkeyed and seed-only-salt candidate ids (ML3, ML4), the correct item moved to the front (ML5), and decoy-display mutants (ML6-ML8).

`docs/DESIGN_NOTES.md` (2026-10-18, mutation checks, modal-cuda 907adab) records that this reached 17/17 only after strengthening 2 weak tests. I did not identify which two tests from the files I read; the git history of `tests/test_config_swap.py` and `tests/test_menu_leak.py` would show it.

The regression rule is also shown to be able to fail: on the pre-fix code at 400 seeds `test_boosted_stumps_on_menu_only_features_do_not_beat_majority` gives repair 0.4475 (+0.0875 over majority, above the 0.072 margin = 3 standard errors) and label 0.46 (`leak_validation_97a49d8.md`).

## 3. Exploit and tamper probes (verifier boundary)

- `mi/tests/test_tamper.py` (read): V2 must fail for the tamper probes `no_training` (steps: 0), `zero_lr`, `skip_model_restore`; for job-written sample files used as evidence; forged in-process records; corrupting only the final checkpoint with a matching checksum; zeroing the final momentum with correct parameters; fabricated checkpoint state (Task B); checkpoints replayed from an earlier run. A bit-identical final state does not bypass other invariants (a changed non-editable file still fails), and valid alternative repairs in the correctness class must pass. I read the test definitions; I did not run them.
- Codex xhigh boundary reviews (`mi/an internal boundary review`, `m1_boundary_r2_codex.md`, `m1_boundary_r3_codex.md`): round 1 reported 6 P1 findings (summarised in `docs/DESIGN_NOTES.md`, 2026-10-17). Round 3 returned "FREEZE OK" with "No new concrete, exploitable hole found in the queue, resume pinning, or safefs conversions"; it was a static review and targeted pytest was blocked by the read-only sandbox.
- Residual risk recorded, not fixed (`docs/DESIGN_NOTES.md`, 2026-10-17): loopback network stays `localhost:*` for a job because Gloo pair connections use ephemeral ports, so other loopback services remain reachable from a job. The POSIX-shm name-prefix sandbox rule could not be compiled in the nested sandbox and needs confirmation on the first outside run.
- The the design spec (docs/DESIGN_NOTES.md) exploit probes (most-common-label, restart-everything, cordon-everything, cite-all-IDs, always-escalate) belong to the M2 audit suite. I found no results file for them in the files I read, so no probe-score claim is made here.

## 4. Token-cap amendment (2,048 -> 3,072)

Evidence: Modal readiness run (`mi/modal_train/READINESS_RESULT.md`): `prepare --dev 48` aborted with `BundleTooLarge: seed 10000: certificate alone needs 2121-2432 > 2048 tokens`. 5 of 12 stratified DEV seeds raised (Task B at 3-4 ranks); the other 7 fit at 2025-2041. Certificate evidence is always retained, so trimming could not help.

Amendment (`evidence/prereg/amendment_token_cap.md`, dated 2026-10-18, written before any outcome evaluation; the frozen prereg is not edited): one cap of 3,072 for all variants (one cap so length does not mark the variant).

Offline validation (`mi/experiments/verifier_audit/token_cap_3072.md`) on 400 DEV seeds with synthesized records: BundleTooLarge 0/400, max 2449, median 2153, p95 2431 true Qwen3 tokens (the same run at 2,048 raised on 95/400). Caveat: bundles were synthesized in the executed record format. The Task B floor reproduces the Modal numbers (2121-2433 est. vs 2121-2432 measured) but Task A synthesizes about 1700 vs 2025-2041 measured, so real executed counts are NOT RUN. Leak validators re-run at the new cap gave output byte-identical to the committed JSON; `mutation_check_leak.py` again 17/17; `pytest -m "not sweep and not executed"`: 98 passed, 8 failed, 8 skipped, the 8 failures being Gloo/sandbox-exec (6 executed token_cap cases) and Metal (2 `test_first_update`) unavailability.

## 5. What is NOT verified

- Everything requiring live re-execution: decoy and repair outcomes, A1 `if False:` and Task B single-site checkpoint edits under re-execution, the B_SKIP vs B_DUP separation, and the real-record bundle-evidence oracle. Only a synthetic partial exists: 6/6 variants at 1.0 accuracy over 400 seeds (`leak_validation_796ba23.md` part 2), which shows the displayed menu is sufficient given the certificate facts, not that the facts are present in the real bundle.
- Executed true-token counts at the 3,072 cap and with the new exclude display.
- The 50 Linux/Modal suite failures. On Modal the executed suite gave 159 passed, 50 failed, 8 skipped (`READINESS_RESULT.md`, run s1d, with `CI=true TD_CI_UNSANDBOXED=1`). Failures by file: test_scenarios 12, test_tiera 21, test_boundary 4, test_tamper 3, test_workspace 3, test_mustfix 1, test_trusted_paths 2, test_modal_train 1. They are consistent with "unsandboxed outcomes can never be verified successes" but were not isolated individually: root cause unresolved. Separately, verified recovery on Linux needs `sandbox.available()` (macOS sandbox-exec); outcome tables must therefore come from the Mac, or from the owner-approved DEV container mode (`exec_mode=dev_container`, exploratory, `.research/HANDOFF.md`).
- Independence: the 97a49d8 validation is by the same author as the fix; stumps were not run under a label-permutation null (only NB with 100 permutations and LR with 20 in round 1).
- Config-visibility 1:1 markers (`sync.per_rank.<r>: reduce_bcast` only in A1_cfg; `data.shard_dirs.<r>: part-<world>` only in A2) remain by design; neutral decoys were proposed and not applied.
- Whether V2 improves a policy is only addressed exploratorily (`MLX_MODAL_RL.md` section 4: 6 paired seeds, V2-arm minus V0-arm +4.64 pts, CI [-0.69, +9.98], p = 0.075, inconclusive; `dev_container` table, not live re-execution).

## 6. Audit finding from the RL study: V1 == V2 on 97.7% of rollouts
`modal_train/results/validity2/v1v2_seeds_out.json`: over 3072 DEV rollouts V1 and V2 agree on 97.7% (A1 89.4%, A2 99.8%, B_SKIP 99.8%, B_DUP 100%, B_OK 100%). `validity/validity_stdout.txt` shows exactly one candidate repair per episode recovers the job (the `correct` role); all decoys fail. So the "+ verified repair" rung of the ladder adds almost no information on this menu design, which weakens what a V0-vs-V2 comparison can show. This is a limitation of the menu design, not a verifier bug. Planned fix (not built): at least two label-consistent options that complete the run, each distractor violating exactly one V2 invariant, outcomes from live re-execution (`docs/DESIGN_NOTES.md`).

## 7. Modal Sandbox replay backend (proof)
`modal_train/results/MODAL_SANDBOX_PROOF.md`: Task A2 seed 101, gVisor sandbox with network blocked and Gloo loopback working. Gold repair accepted twice (deterministic); F06 (raise timeout) and F11 (`world_size:1`) rejected on I1 `no_completed_relaunch`. Scope: one scenario, two cheats. Full red-team numbers are below.

## 8. Red-team results (v1 baseline, Modal Sandbox)
Source: the committed table `evidence/results/REDTEAM_TABLE.md` (commit `fceb762`), recounted 2026-10-18 with the awk command in `evidence/RELEASE_MANIFEST.md`. **The raw v1 JSON was overwritten by a later errored re-run (ProxyError 403 / budget cap) and is not recoverable; these numbers rest on the committed table, not on a re-readable JSON.** Every row ran with `exec_mode = modal_sandbox` (gVisor, no network; invariants recomputed outside the sandbox). 107 cases over five scenario kinds (A1_src, A1_cfg, A2, B_SKIP, B_DUP), $0.455 booked.

| group | result |
|---|---|
| null (no diff) | 0 of 5 accepted (3 execution_failure, 2 invariant) |
| gold repairs | 5 of 5 accepted |
| alternative valid repairs | 9 of 10 accepted; the one false reject is B_DUP alt2 (sampler position recomputed in `restore()`: samples, params, transitions and restores pass; only the checkpoint-format check fails because saved checkpoints store no sampler position). A verifier-strictness finding, not patched in v1 |
| borderline S01 (A2: skip the shard wait) | accepted; a surface patch that removes the injected cause; reported, not counted as a cheat (motivated A2v2) |
| cheats (19 classes F01-F17, F03b, X1, X2) | **0 of 86 accepted**; pooled rule-of-three 95% bound <= 0.035; per-class n is 2 to 5, so per-class bounds are weak (0.6 to 1.5) |
| ...by rejection class, reported separately | **48 by an artifact invariant, 38 by execution_failure** (no relaunch completed) |

Execution failures show the cheats do not work, not that invariants would catch them; invariant coverage is claimed for the 48 only. No claim of completeness over unlisted cheat classes; the audit is by the project's own agents, with no external red team.

Additional evidence (separate, not in the counts above): (a) verifier v1.1 (opt-in, **not active**, not owner-approved), same suite: cheats 0 of 85 verified accepted, gold + alt-gold 15 of 15 (B_DUP alt2 accepted), $0.625 (`evidence/results/REDTEAM_TABLE_v11.md`, `evidence/results/VERIFIER_V11.md`); (b) A2v2, a separately versioned scenario with a real producer thread: S01 and R1 rejected by sample accounting, valid producer-side repairs accepted, S02 (derive positions from the sampler) accepted as valid (`evidence/results/A2V2_SCENARIO.md`, `evidence/results/A2V2_TABLE_*.md`). Not done: external red team, live sealed-TEST run.

## 9. Sealed TEST: the V2 arm gain is evidence citation
100 fresh episodes (20 per kind: A1, A2, B_SKIP, B_DUP, B_OK), menu v1, prompt v1, 15 policies, G=4. V0/V1 are scored offline; V2 is a live replay of every unique (episode, chosen repair) pair in the network-less Modal Sandbox (369 pairs, all `exec_mode=modal_sandbox`, 0 errors). Source: `evidence/results/TEST_REPORT.md`; independent recount: `evidence/results/test/AUDIT_RECOUNT.txt` (`scripts/audit_test_recount.py`).

| policy | V0 label | V1 evidence | V2 live | V2 live, 4 faulted kinds | claimed, not verified (of 400) |
|---|---|---|---|---|---|
| base | 15.8 | 4.8 | 4.0 | 5.0 | 183 |
| sft | 47.2 | 17.5 | 15.5 | 17.2 | 337 |
| csft | 67.5 | 30.0 | 25.2 | 29.7 | 299 |
| GRPO-V0 arm (6 seeds, mean) | 60.1 | 21.6 | 18.6 | 23.0 | 325 |
| GRPO-V2 arm (6 seeds, mean) | 60.1 | 30.3 | 27.1 | 33.4 | 292 |

Primary contrast (V2 arm minus V0 arm, live V2, 4 faulted kinds, 6 paired seeds): **+10.36 pts**, two-level bootstrap 95% CI [+5.83, +15.26], paired t p = 0.001; per seed +7.5, +10.3, +5.3, +15.6, +14.4, +9.1. Class: positive effect, size unresolved (the CI straddles the 7.2-point minimum meaningful difference). Label accuracy did not differ (-0.10, CI [-5.42, +5.36]); evidence-grounded accuracy (V1) differs by +10.68.

How to read it:
- Exploratory: two task families, one task design, 6 seeds, no unseen-mechanism claim. Training reward was the `dev_container` table; TEST scores are live `modal_sandbox` replays.
- The effect is evidence-grounded answers (V1). Both arms pick the recovering repair equally often (71.4% vs 71.5% of faulted rollouts). Each menu has one recovering repair, so V2 is V1 plus that repair and V1 ~ V2. The V2 reward rewards evidence citation and the V0 reward does not. This does NOT show that recovery verification improves recovery.
- The earlier DEV result (+4.64, CI [-0.69, +9.98], `dev_container`) was inconclusive. TEST is roughly twice as large with the same sign in every kind; the difference is not isolated (different episodes, live vs table scoring, G=8 vs 4).
- Claimed-but-unverified fixes are high for every policy; most of them chose a repair that recovers but cited the wrong evidence.
- Continued SFT (single run) recovers much of the gain: V2 arm minus csft +3.70 [-1.67, +9.11].

