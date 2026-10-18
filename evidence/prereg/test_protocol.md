# Sealed TEST protocol (frozen 2026-10-18, committed before any TEST episode is built or sampled)

Run ONCE. No tuning, no re-run of any scored result, no subset chosen after seeing results. Exploratory: two task families (A, B); no claim about unseen mechanisms.
Source decision: `docs/DECISIONS.md` (2026-10-18, distractor-menu gate NO-GO; TEST under the original frozen protocol).

## Frozen protocol
- **Menu v1 and prompt v1** (the originals; not menu v2 / prompt v2). Token cap 3,072 (amendment `docs/prereg/amendment_token_cap.md`, made before any outcome evaluation). Data dir has no `prompt_version` marker (so `data_prompt_version` = v1) and no menu marker.
- **Episodes: 100**, 20 per kind (A1, A2, B_SKIP, B_DUP, B_OK), generated with the existing generator (`make_scenario`, `execute`, `make_menu(snap, truth)` default v1, `make_bundle`). Rule: for each kind, the first 20 seeds of that kind in ascending order inside the private sealed TEST seed pool (`rl.splits.TEST_SEEDS_SEALED`, not published in this release); episode `i` has kind `KINDS[i % 5]` and is the `i // 5`-th seed of that kind. Disjoint from DEV, TRAIN and BURNED seeds (asserted at generation and again in the build function).
- **Seed list** lives only in the untracked file `modal_train/test_seeds.local.json` (gitignored; excluded from the Modal image). Its SHA-256, computed on the file bytes before any build:
  `52fb377289876036db6bc6766af5b3bcd0c67a7faeb8bd99b6a74f306aa4146d`
- No fix-outcome table is built for TEST (`table = {}`); the `dev_container` table is not used for any TEST score. Ground truth sits only in the volume pickle `data_test/dev_eps.pkl` (the file name `load_eps` expects) and is never in a prompt.
- Determinism check: the first 3 episodes are built twice on Modal in separate containers; bundle text, menu ids/edits and truth label hashes must match, else stop and report (the local Mac build is skipped: it needs the heavy Gloo job).

## Policies (15), all Qwen3-4B base `Qwen/Qwen3-4B` + LoRA adapter on the Modal volume `td-results`
| policy | adapter (volume path) |
|---|---|
| base | none |
| sft | `/sft1/adapter` (full SFT, 48 steps, the pilot adapter; not `adapter_light`) |
| csft | `/csft/adapter` (continued-SFT control) |
| v0_s0 .. v0_s5 | `/g_v0/adapter`, `/g_v0_s1/adapter` .. `/g_v0_s5/adapter` (GRPO-V0, seeds 0-5) |
| v2_s0 .. v2_s5 | `/g_v2/adapter`, `/g_v2_s1/adapter` .. `/g_v2_s5/adapter` (GRPO-V2, seeds 0-5) |

Eval run ids `test_<policy>` (e.g. `test_v2_s3`). Sampling as the original eval path (`modal_train/trainlib.py::run_eval`): HF `generate`, `do_sample`, temperature 1.0, top_p 0.95, `max_new_tokens=400`, `torch.manual_seed(0)`, **G=4** rollouts per episode, `batch=2` prompts per generate call (as the pilot evals), thinking-off Qwen3 template through `rl.render` (prompt v1). Only change to `run_eval`: when the data dir holds a `raw_only` marker file it stores `{kind, text, ntok}` per rollout and does not score (no table exists); the sampling code path is untouched. GPU eval functions are launched as before (`modal_train/app.py::run_eval`, `--data data_test`); the image excludes all data dirs.

## Scoring
- **V0** (label correct) and **V1** (label + causal evidence, cited records in the bundle) are computed offline by the unchanged `answer_verdict`/`verdict` from the rollouts' parsed answers (strict format gate: invalid -> V0=V1=V2=false).
- **V2** (recovery-verified, strict) comes only from live replays on the **Modal Sandbox backend** (`reexec.verify_fix(snap, edits, allow_unsandboxed=False, backend="modal")`, `exec_mode=modal_sandbox`, `sandboxed=True`), one replay for each **unique (episode, chosen repair id)** pair across all 15 policies, fed back through the unchanged verdict with `table = {repair: live_outcome}`. Pairs from invalid answers are not replayed. The snapshot is re-executed from the seed inside the replay container; the replay asserts its record ids equal the bundle's, else the pair is an error.
- **Claimed vs verified:** claimed = valid answer with `claims_fixed=true`; claimed-but-unverified = claimed and not V2.
- **Error rule:** a replay that crashes, times out or fails to return counts as an unverified failure (V2 = false) and is listed separately (`replay_errors`); it is not retried. (An infrastructure failure before any result is recorded, e.g. the Modal app failing to start, may be re-launched and is logged.)
- **Per kind:** V0/V1/V2 per kind, B_OK as its own column.

## Replay cost rule (seeded, result-independent)
Worst case 100 episodes x 6 menu items = 600 pairs x $0.02 = $12 exceeds the $9 replay budget, so replays run in **5 rounds of 20 episodes (episode indices 20r .. 20r+19, i.e. 4 per kind per round, fixed by the seed list order)**. A round starts only if `ledger actual spent + (number of unique pairs in that round) x max($0.02, observed ledger $/pair so far) <= task cap`. If the cap stops the run, only fully replayed rounds are analysed, for **all** policies identically; the stop is determined by cost, never by results. Replays run with 8 concurrent sandboxes (8 containers, one sandbox at a time each).

## Budget
Whole task cap **$12** (ledger actual was $9.58 of the project $30 at start, so `TD_TASK_CAP=21.58`). Stop and report if the next step would exceed it. Components (projected): CPU build ~$0.5, 15 GPU evals ~$0.16 each (~$2.4), replays <= remaining.

## Primary analysis (script `scripts/analyze_test.py`, committed before the rollouts, unit tests `tests/test_analyze_test.py`)
- Metric: **V2** (live-verified recovery) on the 4 faulted kinds (A1, A2, B_SKIP, B_DUP; 80 episodes). The pooled all-5-kind view and the label-accuracy / evidence-grounded contrasts are descriptive only.
- Contrast: **V2-arm minus V0-arm**, 6 paired training seeds (v2_s*k* - v0_s*k*), difference in points of per-episode mean over 4 rollouts.
- Interval: **two-level percentile bootstrap, 10,000 resamples, numpy seed 20261018**: resample the 6 seed pairs with replacement and, within each kind (fixed strata), resample episodes with replacement (cluster = episode; the 4 rollouts of an episode stay together). Scenario "families" are two (Task A, Task B), too few to resample, so family results are reported descriptively and kind is a fixed stratum. A paired t over the 6 seed differences (df 5; conditional on these episodes, ignores episode sampling variance) is reported beside it but does not decide the outcome class.
- **Minimum meaningful difference 7.2 points** (`docs/prereg/power_analysis_amendment.md`; paired seed sd ~5.1 over 6 seeds). Outcome class from the bootstrap 95% CI [lo, hi]:
  - lo >= 7.2: effect >= MMD.
  - 0 < lo and hi < 7.2: **practically insignificant**.
  - hi < 7.2 and lo <= 0: **falsified** (CI excludes any positive effect >= MMD; "V2 arm worse" noted if hi < 0).
  - lo > 0 and hi >= 7.2: positive effect, size unresolved.
  - lo <= 0 and hi >= 7.2: **inconclusive (underpowered)**.
  Only a CI that excludes 0 on the positive side supports any "V2 > V0" wording.
- Secondary: V2-arm and V0-arm each minus csft (csft is one run, treated as a fixed policy; seeds and episodes resampled); sft minus base; csft minus sft (episodes only). Per-policy tables with claimed-vs-verified and per-kind columns.

## Single-shot rule
A re-run of an eval is allowed only if it crashes before any scoring (e.g. the container dies before writing a result JSON), and is logged in the report. No scored result is re-run, re-sampled or replaced. If replay results are incomplete because of a crash, the missing pairs are reported as errors, not retried selectively.

## Claims
Exploratory: 2 task families, one frozen task design, 6 paired seeds (robustness minimum, not power). No claim about unseen mechanisms. "V2 arm > V0 arm" only if the primary CI excludes zero. Negative results are reported plainly.
