# TrainDoctor v1 release manifest (2026-10-18)

One row per headline claim: the file that holds the number, the command that produced it, the commit, and the `exec_mode`.
Commit hashes refer to the private development history and do not exist in this repository. Paths are relative to this repository (the repo root) unless stated. Copies of the gitignored
originals are in `docs/results/` (hashes in `docs/results/SHA256SUMS`). This repository is public at github.com/TarangKhanna/traindoctor-env (v0.1 research preview). Nothing has been published to the Hugging Face or OpenEnv hubs, and no upstream issue or PR has been filed.

`exec_mode` legend: **modal_sandbox** = repair executed in a network-less gVisor Modal Sandbox and invariants recomputed on the trusted side
(the verified path). **dev_container** = trusted pinned code in local Docker; EXPLORATORY training-reward view only, never a verified result.

Base code commit for the audit and sandbox rows: `modal-infra` `fceb762` (red-team results) on top of `fc370e1`/`cee1b09`/`ba04949` (Modal backend) and `9563698` (suite).
Verifier v1.1 code: `c8e5581`. 

## Claims

| # | claim | number | source file | command / script | commit | exec_mode |
|---|---|---|---|---|---|---|
| 1 | Red-team v1, null diff | 0 of 5 accepted (3 execution_failure, 2 invariant) | `docs/REDTEAM_TABLE.md` | `TD_KEY=<any hex> TD_MODAL_SITE=<site-packages containing modal> PYTHONPATH=. .venv/bin/python scripts/redteam_suite.py --run modal` (default `--verifier v1`, `--cap-usd 1.5`; allowed hosts: Modal API only) | `fceb762` | modal_sandbox |
| 2 | Red-team v1, gold repairs | 5 of 5 accepted | same | same | `fceb762` | modal_sandbox |
| 3 | Red-team v1, alternative valid repairs | 9 of 10 accepted; the one false reject is **B_DUP alt2** (sampler position recomputed in `restore()`): samples, params, transitions and restores pass, only the checkpoint-format check fails because saved checkpoints carry no sampler position. A verifier-strictness finding, not patched in v1 | same | same | `fceb762` | modal_sandbox |
| 4 | Red-team v1, borderline **S01** (A2: skip the shard wait) | **accepted** in A2 v1. This is a surface patch that removes the injected cause; it is reported, not counted as a cheat. It is the reason A2v2 exists (row 12) | same | same | `fceb762` | modal_sandbox |
| 5 | Red-team v1, cheats | **0 of 86 accepted**; 95% rule-of-three bound <= 0.035 pooled; per-class bounds up to 1.5 (n=2..5 per class) | same | same | `fceb762` | modal_sandbox |
| 5a | ...rejection class split, reported separately | **48 rejected by an artifact invariant, 38 by execution_failure** (the job never completed a relaunch). *Execution failures show the cheats do not work, not that invariants would catch them; invariant coverage is claimed for the 48 only.* | same | recount below | `fceb762` | modal_sandbox |
| 6 | Cost of red-team v1 | $0.455 booked | `docs/REDTEAM_TABLE.md` run notes; `docs/DESIGN_NOTES.md` | ledger rows `fn == redteam_modal` | `fceb762` | n/a |
| 7 | Red-team v1.1 (opt-in verifier, **NOT active**, owner has not approved) | 107 cases: cheats 0 of 85 verified accepted (48 invariant, 37 exec, 1 infra error), null 0/5, gold 5/5, alt-gold 10/10 (B_DUP alt2 now accepted), S01 accepted; $0.625 | `docs/results/redteam_modal_v11.json`, `docs/results/REDTEAM_TABLE_v11.md`, `docs/VERIFIER_V11.md` | same command plus `--verifier v1.1` (outputs get `_v11`) | `c8e5581` (opt-in verifier v1.1 code); table cited in `docs/VERIFIER_V11.md` | modal_sandbox |
| 8 | Six-seed V2-arm minus V0-arm | **+4.64 pts, sd 5.08, t=2.24 (df 5), p=0.075, 95% CI [-0.69, +9.98]; INCONCLUSIVE**. Per seed +1.8, +5.2, +6.0, +10.2, +8.6, -3.9. On the V0 metric -0.13 (p=0.95). The preregistered minimum meaningful difference (20 pts) was never in reach | `docs/results/SEEDS_3_5_REPORT.md`, `docs/results/exploratory_v2/*_EXPLORATORY.json` | `PYTHONPATH=. python -m modal_train.results.analyze_exploratory <run>` per run (script is in the gitignored `modal_train/results/`); the mean/sd/t/CI recomputed 2026-10-18 from the saved `v2_exploratory` fields: 4.644, 5.083, 2.238, [-0.690, +9.978] | code `76310e2`/`3f52537` lineage (`modal_train/{core,trainlib,app}.py`); results gitignored | **dev_container, EXPLORATORY** (old prompt, 48 DEV episodes x G=8, one run per arm per seed) |
| 9 | Continued-SFT control | expl. V2: SFT 0.154, continued SFT 0.224, GRPO-V0 0.245, GRPO-V2 0.263 (seed 0). Continued SFT recovers most of the GRPO gain; 6-seed arms vs the single csft run: V0 arm -3.2 pts (p=0.08), V2 arm +1.5 (p=0.44), treating csft as a constant (optimistic) | `docs/results/CONTROLS_REPORT.md`, `docs/results/CONTROLS_SUMMARY.json`, `docs/results/SEEDS_3_5_REPORT.md` | `PYTHONPATH=. python -m modal_train.results.analyze_controls` | as row 8 | dev_container, EXPLORATORY |
| 9a | Pilot (seed 0) | base 0.26% -> SFT 15.4% -> GRPO-V0 24.5% / GRPO-V2 26.3% | `docs/results/PILOT_REPORT.md` | commands listed in the report (`run_sft`, `run_eval`, `run_grpo` via `modal_train/run.sh`) | as row 8 | dev_container, EXPLORATORY |
| 10 | V1 == V2 on rollouts | 97.7% of 3072 DEV rollouts (A1 89.4%, others >= 99.8%); V1 passes where V2 fails on 70 | `docs/results/v1v2_seeds_out.json` (+ `v1v2_seeds_stdout.txt`) | `modal_train/results/validity2/v1v2_seeds.py` (needs gitignored `modal_train/cache/dev_eps.pkl`; not re-run, output file inspected) | as row 8 | dev_container, EXPLORATORY |
| 11 | Modal Sandbox proof | Task A2 seed 101: network blocked, loopback Gloo works, gold accepted twice, F06 and F11 rejected (I1); ~$0.07 | `docs/results/MODAL_SANDBOX_PROOF.md`, `docs/results/modal_sandbox_proof.json` | `scripts/modal_sandbox_proof.py A2` | `fc370e1` | modal_sandbox (one scenario, two cheats) |
| 12 | A2v2 (separately versioned scenario, **extra evidence, not part of the headline counts**) | 11 cases on one scenario (world 3, seed 101): null, gold, alt1, alt2 and S02 behave as expected; S01 (skip the wait) and R1 are rejected by sample accounting (gaps=53, dups=53, I3), the v1 gap is closed for that patch class. **S02** (remove the wait and derive positions from the sampler) is *accepted* and judged a valid repair. F06 errored once (sandbox lifetime under parallel load), re-run alone and rejected (execution_failure). Cheats in A2v2: 3 verified rejected (F11, F04 exec; X2 invariant) plus F06 alone: 4 rejected of 4, n too small for bounds | `docs/A2V2_SCENARIO.md`, `docs/results/A2V2_TABLE_v2a.md`, `docs/results/A2V2_TABLE_v2b_f06.md`, `docs/results/redteam_a2v2_v2a.json`, `docs/results/redteam_a2v2_v2b_f06.json` | `scripts/redteam_a2v2.py` (runs v2a and v2b_f06); tests `tests/test_a2v2.py` | `6c56f41` | modal_sandbox |
| 13 | Menu-leak audit | menu-only predictors <= majority after the fix; 17 of 17 in-memory mutants of the leak tests caught (same-author validation) | `docs/writeups/VERIFIER_AUDIT.md`; `tests/test_menu_leak.py` | `pytest tests/test_menu_leak.py` (6 pass, below); the mutant run is `scripts/mutation_check_leak.py` (not re-run here) | see `docs/DESIGN_NOTES.md` | n/a |
| 14 | Spend | **$8.04 actual** (418 `kind == "actual"` rows, snapshot 2026-10-18): eval 2.99, grpo 2.15, **redteam_modal 2.27 (includes the errored 403 run and the isolation/smoke runs; ledger counts it, the v1 baseline itself was $0.455)**, cpu_s1 0.30, sft 0.24, modal_sandbox_proof 0.07, synth 0.01. `projected` rows are reservations. Cap $30 incl. $2 reserve. The ledger is still being appended by running jobs | `docs/results/LEDGER_ACTUAL_SUMMARY.txt` (derived from gitignored `modal_train/ledger.jsonl`, sha256 prefix at snapshot eb74daacc877 was for 455 lines and has since grown) | inline python summing `usd` over `kind == "actual"` rows by `fn` | n/a | provider-side billing not itemized or checked |

| 15 | Sealed TEST, V2 arm minus V0 arm (live-verified V2, 4 faulted kinds, 6 paired seeds; EXPLORATORY) | **+10.36 pts**, bootstrap 95% CI [+5.83, +15.26], paired t p=0.001; class: positive effect, size unresolved (MMD 7.2). Label accuracy -0.10; V1 evidence +10.68. Both arms choose the recovering repair equally often (71.4% vs 71.5%), so the effect is evidence citation, not recovery. 15 policies, 100 episodes, 369 live replays | `docs/results/TEST_REPORT.md`, `docs/results/test/` | `scripts/analyze_test.py` (frozen `e2e2560`); independent recount `scripts/audit_test_recount.py` -> `docs/results/test/AUDIT_RECOUNT.txt` | `e2e2560`..`3a802f6` | modal_sandbox |
| 16 | TEST per-policy V2 live (all 5 kinds) | base 4.0, sft 15.5, csft 25.2, V0 arm mean 18.6, V2 arm mean 27.1; claimed-not-verified 183 to 343 of about 400 per policy (csft 299) | same | same | same | modal_sandbox |

## Recounting the red-team numbers

The v1 baseline raw JSON is **lost** (see next section). The recount is from the committed table, which is the only surviving record:

```bash
awk -F'|' 'NF==11 && $2!~/ID|---/ {id=$2; gsub(/ /,"",id); acc=$4; gsub(/ /,"",acc); cl=$5; gsub(/ /,"",cl);
  r=(id=="null")?"null":(id=="gold")?"gold":(id~/^alt/)?"alt":(id~/^S[0-9]/)?"borderline":"cheat"; n[r" "acc" "cl]++}
  END{for(k in n)print k, n[k]}' docs/REDTEAM_TABLE.md | sort
```

Output on 2026-10-18 (107 rows): `alt no invariant 1`, `alt yes accepted 9`, `borderline yes accepted 1`, `cheat no execution_failure 38`,
`cheat no invariant 48`, `gold yes accepted 5`, `null no execution_failure 3`, `null no invariant 2`. That is null 0/5 accepted, gold 5/5,
alt-gold 9/10, cheats 0/86 (48 + 38). The same command on `docs/results/REDTEAM_TABLE_v11.md` gives alt 10/10, cheats 48 invariant + 37 execution_failure + 1 ERROR.
Every row in both tables has `exec_mode = modal_sandbox`.

## Honesty notes (read before quoting any number)

1. **Raw v1 red-team JSON was overwritten.** `modal_train/results/redteam_modal.json` was rewritten by a later re-run (run `modal`, $1.331,
   all 107 rows `rejection_class == error`: `ProxyError: 403 Forbidden` on A1_src and `BudgetExceeded` after). It is kept, quarantined, as
   `docs/results/redteam_modal_ERRORED_403_run.json` (original renamed to `modal_train/results/redteam_modal_ERRORED_403_run.json`). It contains
   **no verification result** and must not be cited. No recoverable copy of the good run exists (searched the worktree, `$TMPDIR`/`/tmp/claude-501/rt_*`,
   tdrun scratch copies, git objects; the file was never tracked). The baseline numbers therefore rest on `docs/REDTEAM_TABLE.md` (committed in `fceb762`) and
   the agents' reports, not on a JSON that can be re-read. The two isolation runs (F06 hangs for real; F11 fails I6 on A1 but stalls on A2, cause not isolated)
   survive as `docs/results/redteam_isolate.json` and `redteam_isolate_long.json`.
2. 38 of the 86 cheat rejections are execution failures. They show that the cheat does not produce a working job, not that an invariant would catch it.
3. Cheats rejected by a crash or timeout are fail-closed; `error` rows are excluded from the rule-of-three denominators in every table.
4. The red-team has 19 cheat classes and 107 cases on five scenario kinds; per-class n is 2 to 5, so per-class bounds are weak. No claim of completeness over unlisted cheat classes. The audit is by the project's own agents plus reviews; there is no external red team.
5. B_DUP alt2 is rejected by v1 and accepted by v1.1; v1.1 is opt-in, not approved, and not used for any number above except row 7.
6. All learning results (rows 8-10) are `dev_container` EXPLORATORY: old prompt (presupposes a fault; B_OK ~0%), DEV is a tuning set, fix table not live re-execution, one run per arm per seed, sealed TEST was run once afterwards (rows 15-16). The headline requirement (live sandboxed re-execution on sealed TEST, once) has **not** been met.
7. Frontier-agent (M1.5) numbers are not used: the results file is not on these branches.

## Not done / pending

- Shared-prefix (mlxrl) timing: pending, file that will fill it: `docs/SHARED_PREFIX_STATUS.md` on the mlxrl `sharedprefix-grpo` branch (status NOT READY).
- v1.1 verifier, distractor-menu gate, prompt-v2 retraining: pending, will be filled by `docs/VERIFIER_V11.md` (owner approval), `modal_train/results/gate_analysis.py` output, and a new `modal_train/results/*_EXPLORATORY.json` set.
- Sealed TEST: done once (rows 15-16). Interpretation: exploratory; the V2-arm gain is evidence citation, both arms pick the recovering repair equally often, so it does not show that recovery verification improves recovery; DEV (+4.64) was inconclusive; the CI straddles the 7.2-point MMD; no unseen-mechanism claim.
