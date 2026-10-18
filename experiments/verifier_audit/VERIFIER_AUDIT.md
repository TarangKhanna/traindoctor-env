# Verifier audit: what was checked, what was caught, what is not verified

Scope: the Tier A prompt bundle and menu (can a model read the answer off the menu alone?), the V0/V1/V2 verifier and its tamper boundary, and the tests that guard them. The audit was done by the project's author with automated reviews; no external red team has run. The leak-validation scripts and their raw outputs are archived privately and are not published; the checks below can be re-run with `audit/menu_leak.py` and the tests named in each section.

Verifiers: V0 = label correct; V1 = V0 + cited evidence observed and relevant; V2 = V1 + verified repair by re-execution with recovery-correctness invariants (committed sample accounting, restored state versus reference, lost work within one checkpoint interval, rank consistency, tamper checks) + safety gates. See [`docs/VERIFICATION.md`](../../docs/VERIFICATION.md).

## 1. Menu leak (can the menu alone predict the answer?)

Setup: 400 development seeds (10000..10399), 5-fold cross-validation across seeds, Wilson 95% intervals, CPU only; majority class 0.360 for repair slot and label. Regression guard: `python -m audit.menu_leak` and `tests/test_menu_leak.py`.

- First validation: failed for one isolated cause. Boosted stumps on menu-only features reached 0.495 (repair slot) and 0.485 (label) against 0.360, because two menu items drew context lines from the buggy job text while the others used clean text. A second finding: candidate ids were a public hash of (seed, role), which recovered the seed in 100/100 menus.
- Fix (`traindoctor_env/bundle.py`): every menu item context is rendered from the clean text in one neutral form; candidate ids are keyed with a per-scenario secret derived from `TD_KEY`.
- Re-validation: menu-only out-of-fold accuracy for boosted stumps is 0.333 [0.288, 0.380] (repair slot) and 0.300 [0.257, 0.347] (label); counterfactual groups (200 groups x 6 variants, chance 0.333) give 0.273 and 0.266; the old id formula recovers the seed from 0/100 menus. This validation was run by the same author as the fix.
- Known and kept, not a menu leak: the bundle config alone reveals the config-level collective bug (`reduce_bcast`) and the A2 shard directory (`part-<world>`). A config-only rule scores 1.000 label accuracy on config-obvious training scenarios (n=124) and 0.533 on evidence-ambiguous ones (n=276), so results are reported overall and by those two strata ([`evidence/prereg/strata_v0_v2.md`](../../evidence/prereg/strata_v0_v2.md)). `tests/test_config_swap.py` guards that no config key encodes a label.

## 2. Mutation check of the leak tests

`python scripts/mutation_check_leak.py` applies 17 in-memory mutants (library code on disk is never changed) and requires each leak or config-visibility test to pass on the real code and fail on the mutant. Output: [`mutation_check_leak.json`](mutation_check_leak.json): 12 tests pass on real code; 17 of 17 mutants detected. The two weakest tests were strengthened before this count was reached.

## 3. Tamper probes

`tests/test_tamper.py`: V2 must fail for no training (`steps: 0`), zero learning rate, skipped model restore, job-written sample files used as evidence, forged in-process records, a corrupted final checkpoint with a matching checksum, a zeroed final momentum with correct parameters, fabricated checkpoint state, and checkpoints replayed from an earlier run. A bit-identical final state does not bypass other invariants, and valid alternative repairs must pass. Residual risk: loopback networking stays open to `localhost:*` inside a job because Gloo uses ephemeral ports, so other loopback services remain reachable from a job.

## 4. Token cap (2,048 to 3,072)

A readiness run aborted because the Task B certificate alone needs 2,121 to 2,432 tokens, above the original 2,048 cap, and certificate evidence is never trimmed. The cap was raised to one value for all variants ([`evidence/prereg/amendment_token_cap.md`](../../evidence/prereg/amendment_token_cap.md)), before any outcome evaluation. On 400 development seeds with synthesized records the maximum is 2,449 true Qwen3 tokens; real executed counts at the new cap were not measured.

## 5. Red team (v1 baseline, Modal Sandbox)

Source: [`evidence/results/REDTEAM_TABLE.md`](../../evidence/results/REDTEAM_TABLE.md), recounted with the command in [`evidence/README.md`](../../evidence/README.md). The raw v1 JSON was overwritten by a later errored run; the numbers rest on the committed table. All 107 cases ran with `exec_mode = modal_sandbox` over five scenario kinds.

| group | result |
|---|---|
| null (no diff) | 0 of 5 accepted (3 execution failure, 2 invariant) |
| reference repairs | 5 of 5 accepted |
| alternative valid repairs | 9 of 10 accepted; the false reject is B_DUP alt2 (sampler position recomputed in `restore()`), rejected only by the checkpoint-format check |
| borderline S01 (A2: skip the shard wait) | accepted; a surface patch that removes the injected cause, not counted as a cheat; motivated A2v2 |
| cheats (19 classes) | **0 of 86 accepted**; pooled rule-of-three 95% bound <= 0.035; per-class n is 2 to 5 |
| by rejection class | **48 by an artifact invariant, 38 by execution failure** |

Execution failures show the cheats do not work, not that invariants would catch them; invariant coverage is claimed for the 48 only. Verifier v1.1 (opt-in, not active) accepts the B_DUP alt2 repair and still rejects all verified cheats (0 of 85): [`REDTEAM_TABLE_v11.md`](../../evidence/results/REDTEAM_TABLE_v11.md), [`VERIFIER_V11.md`](../../evidence/results/VERIFIER_V11.md). A2v2, a separately versioned scenario with a real producer thread, is in [`A2V2_SCENARIO.md`](../../evidence/results/A2V2_SCENARIO.md); the Modal Sandbox properties (network blocked, loopback Gloo works) are in [`MODAL_SANDBOX_PROOF.md`](../../evidence/results/MODAL_SANDBOX_PROOF.md).

## 6. V1 equals V2 on 97.7% of development rollouts

[`evidence/results/v1v2_seeds_out.json`](../../evidence/results/v1v2_seeds_out.json): over 3072 development rollouts V1 and V2 agree on 97.7% (A1 89.4%, A2 99.8%, B_SKIP 99.8%, B_DUP 100%, B_OK 100%). Exactly one candidate repair per episode recovers the job, so the "verified repair" rung adds little information on this menu design. This is a limitation of the menu design, not a verifier bug.

## 7. Sealed TEST

100 fresh episodes (20 per kind), menu v1, prompt v1, 15 policies, G=4. V0 and V1 are scored offline; V2 is a live replay of every unique (episode, chosen repair) pair in the network-less Modal Sandbox (369 pairs, 0 errors). Source: [`TEST_REPORT.md`](../../evidence/results/TEST_REPORT.md); independent recount: [`AUDIT_RECOUNT.txt`](../../evidence/results/test/AUDIT_RECOUNT.txt). The primary contrast (V2 arm minus V0 arm, live V2, four faulted kinds, six paired seeds) is +10.36 points, bootstrap 95% interval [+5.83, +15.26], and comes from evidence citation: both arms pick the recovering repair equally often (71.4% vs 71.5%). The result is exploratory (two task families, no unseen-mechanism claim).

## 8. Not verified

- Live re-execution of decoy repairs, and executed true-token counts at the 3,072 cap.
- Independence of the leak validation (same author as the fix); the stumps were not run under a label-permutation null.
- Linux executed-suite results: see [`docs/TESTING.md`](../../docs/TESTING.md).
