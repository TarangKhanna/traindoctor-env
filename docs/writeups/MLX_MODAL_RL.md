# Local MLX RL measurements and the Modal training pipeline

Status: infrastructure measurements plus one exploratory real-episode SFT/GRPO study (section 4). **All RL numbers are EXPLORATORY: DEV split, `dev_container` fix table (not live sandboxed re-execution), old prompt, one run per arm per seed, 6 paired seeds; the V2-arm vs V0-arm comparison is INCONCLUSIVE.** Qwen3-0.6B numbers are plumbing only and never results (the project rules rule 8). All local timings were taken on a memory-pressured, CPU-contended machine unless stated (`docs/NOTES.md` header, `docs/m0/smoke.md`).

Sources: `docs/NOTES.md` (M0), `docs/m0/runs/*.json`, `docs/LEARNINGS.md`, `docs/PUBLISH.md`; Modal: `modal_train/{README.md,SMOKE_RESULT.md,READINESS_RESULT.md,rates.json,ledger.jsonl}`; `.research/HANDOFF.md`.

## 1. Pins

Python 3.12.9, uv 0.6.6, macOS 26.5.2, MacBook Air M4 (fanless), 32 GB. mlx 0.31.2, mlx-lm 0.31.3, torch 2.14.1 (CPU), inspect-ai 0.3.281, openenv 0.9.0. mlxrl (`inductive-mlxrl`) git `25054948b19461569e25f8b657ca041b4bc7740b` (= v0.2.2). Models (4-bit, group 64): Qwen3-4B rev `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, Qwen3-0.6B rev `73e3e38d981303bc594367cd910ea6eb48349da8` (`docs/NOTES.md` section 1). mlx and mlx-lm are pinned down from latest (0.32.4 / 0.32.2) because on those mlxrl's prefix-cached rollouts fail with `AttributeError ... 'meta_state'` and batched ones with `too many values to unpack`; naive rollout works on both (`docs/NOTES.md` section 1, `docs/LEARNINGS.md`).

## 2. MLX RL measurements on the M4 Air (M0)

Plumbing and feasibility, mlxrl via its Python API (Tier A path).

| item | result | source |
|---|---|---|
| Executable fault proof (4-rank Gloo, CPU) | PASS: rank 2 calls `broadcast` while others `all_reduce` at step 3; all 4 ranks time out, last completed step 2, exit codes 1,1,1,1, 9.3-10.2 s, identical across 5 runs; fixed run 20 steps, exit 0, identical weight hash on all ranks, 3/3 runs | `docs/NOTES.md` section 8; `docs/m0/fault_proof.md` |
| 0.6B GRPO plumbing | loss finite every step; step 1: 196/392 LoRA tensors changed (all `lora_b`), after 4 steps 392/392, frozen 0/704 changed. First run had all-equal rewards: loss 0.0, nothing changed | `docs/m0/runs/p06_grpo_batch.summary.json` |
| GiGPO example (0.6B) | 5 steps, losses finite, 35.6 s incl. load, mlx peak 2.85 GB; does not learn in 5 steps | `docs/NOTES.md` section 3 |
| Qwen3-4B, 71-token prompt, 400 gen tokens, 10 steps, G4 no checkpointing | 85.8 s/step, peak 26.4 GB | `docs/m0/runs/p4b_g4.summary.json` |
| same, G8 checkpointing + micro-batch 2 | 216.3 s/step, peak 9.2 GB; 504/504 LoRA tensors changed, 0/904 frozen | `docs/m0/runs/p4b_g8_ckpt.summary.json` |
| same, G8 no checkpointing | aborted after 2 steps, 187 s/step, peak 35.3 GB | `docs/m0/runs/p4b_g8_nockpt_ABORTED.*` |
| 300-step extrapolation (contended) | G4 no-ckpt about 6.5 h but over the 24 GB budget; G8 ckpt+micro2 about 16 h | `docs/NOTES.md` section 3 |
| 0.6B at a real-shape prompt (3505 tokens), G4, no ckpt, stock | backward 373 s, mlx peak 86.4 GB (does not fit) | `p06_realshape_nomicro_3505tok` |
| same with completion-only projection (`slice`) | no ckpt: peak 51.7 GB, backward 141 s; with ckpt: 62.8 s/step, peak 6.3 GB. No controlled stock run at 2641 tokens | `p06_rs_g4_slice_noc`, `p06_rs_g4_slice_ck` |
| Rollout modes, 4B, G4, 2641-token prompt | naive 107-149 s, batch 60-72 s, prefix 52-63 s; 4/4 identical completions at T=0.7 and 1.0 (weak evidence) | `docs/LEARNINGS.md`; `eq4b_rollout_2500_T07/T10` |
| Prefill, Qwen3-4B-4bit 2641 tokens | 23 s (115 tok/s, contended) | `docs/m0/smoke.md` |

Equivalence of the completion-only projection: `slice` was bit-identical (loss, grads) on a synthetic 913-token case; `shared` fp32 grad relative L2 1.9e-5 but bf16 0.246; `shared` with micro-batch gave wrong gradients (rel 1.13), uninvestigated (`docs/NOTES.md` section 3, `docs/m0/logprob_projection.md`). Gradient checkpointing alone gave exactly 0.0 difference in fp32 tests; micro-batching is not bit-exact (fp32 grad rel L2 about 1e-4; bf16 about 0.03).

**Not measured:** sustained throughput over >= 20 minutes (thermal; mid-run 4B speed rose from about 13 to 40 tok/s when CPU load dropped, not isolated); 4B at the real Tier A prompt shape; 8B training (weights on disk, not run); mlx-lm serving of 8B/14B; SGLang MLX backend. `docs/NOTES.md` section 4 states no go/no-go claim for 4B at the real shape: stock mlxrl cannot fit it, so any 4B Tier A GO depends on adopting a completion-only projection (the basis of PR A, see `BENCHMARKS.md`).

## 3. Modal pipeline

Why Modal: the Mac cannot run the real-shape 4B GRPO cheaply (section 2), so a bounded CUDA path was ported (`modal_train/`, isolated from the core env; core packages never import Modal). Pins chosen 2026-10-18: modal client 1.3.5, torch 2.14.1, transformers 5.20.1, trl 1.16.0, peft 0.21.2, accelerate 1.15.0. Flow: local `prepare` (real episodes, DEV 48 + TRAIN 64) -> volume upload -> 0.6B smoke -> SFT -> GRPO -> eval. Spend guard: every launch writes a projected row to `modal_train/ledger.jsonl`, refuses above $30 total, and replaces it with actual container seconds. Rates verified against `modal billing rates` on 2026-10-18 (`rates.json`): L4 $0.80/h, A10G $1.10, L40S $1.95, A100-40GB $2.10, A100-80GB $2.50, H100 $3.95, CPU $0.14/h.

### 3a. CUDA plumbing smoke (synthetic data)
`SMOKE_RESULT.md`: dataset `synthetic_smoke` (16 hand-made Tier-A-shaped prompts, stub reward); Qwen/Qwen3-0.6B bf16 LoRA r16 on one NVIDIA L4 (23.7 GB).

| item | value |
|---|---|
| container seconds | 47.96 (includes model download and load; image build 56 s not billed to GPU) |
| SFT, 2 steps, batch 4 | wall 5.85 s, peak 1.66 GB, 1359 tokens, 232 tok/s |
| GRPO, 1 update, group 4, max_new 128 | wall 3.92 s, peak 2.99 GB, rollout+score 2.72 s, optimizer 0.036 s, 688 tokens |
| cost | ledger $0.0107 of task cap $1.50 (projected worst case $0.20) |

**Result that matters:** after SFT on 8 rows all 4 sampled completions were identical, so reward std was 0, loss 0 and the advantage was zero. The gradient path executed, but **no learning signal existed on this data, so there is no gradient claim.** The SFT-to-GRPO adapter handoff (`PeftModel.from_pretrained(is_trainable=True)` into `GRPOTrainer`) worked, and no TRL 1.16 / transformers 5.20.1 / peft 0.21.2 API breakage was hit. The Modal-side billing page was not checked for this run, so the provider charge is unconfirmed.

### 3b. S1 readiness run (CPU Gloo on Modal), then stopped
`READINESS_RESULT.md`: Modal CPU container (gVisor, 20 visible CPUs, Python 3.12.10). Gloo/loopback works: a 2-rank `all_reduce` probe returned sum 3.0 on both ranks. Executed suite on Linux (with `CI=true TD_CI_UNSANDBOXED=1`): 159 passed, 50 failed, 8 skipped; the 50 failures are not individually isolated (see `VERIFIER_AUDIT.md`). `prepare --dev 48` aborted with `BundleTooLarge` (certificate alone needs 2121-2432 tokens > 2048 for Task B at 3-4 ranks), which led to the cap amendment to 3,072 (`docs/prereg/amendment_token_cap.md`). A second, independent blocker: verified recovery on Linux needs macOS `sandbox-exec`, so every outcome-table entry produced unsandboxed is invalid. The run stopped at the S1 gate; S2 (GPU) was not launched.

Spend: CPU only. Ledger $0.30; provider billing summary metered 0.89 -> 1.19, billed $0.00 (credits). Total ledger so far: $0.30 + $0.0107 = about $0.31 against the $30 cap. No running apps afterwards.

### 3c. Owner decisions in force (`.research/HANDOFF.md`)
Training-only DEV container mode approved (image `td-dev:1`, read-only worktree, offline, non-root); outcomes from it are labeled `exec_mode=dev_container` and are exploratory, distinct from independently verified recovery. Pilot budget $6, spend cap $30 including a $2 reserve. Pilot plan: base eval on frozen episodes -> oracle SFT -> GRPO on real episodes with verified rewards -> evaluate base, SFT-only and SFT+GRPO on the same held-out cases. If GRPO shows zero reward variance, invalid outputs, or no meaningful updates, stop and diagnose before spending GPU time. The HANDOFF records that `td_dev_token_cap` and the other `td_dev_*` specs had not produced results when that session stopped; they were run later (section 4).

## 4. Real-episode SFT and GRPO study (EXPLORATORY)

Every number here comes from `modal_train/results/` (paths relative to ``). Labels that apply to all of it: Qwen3-4B, LoRA r16; DEV split (48 episodes x G=8 = 384 samples per row), which became a tuning set; the fix-outcome table was produced in `dev_container` mode (`exec_mode=dev_container`), so "expl. V2" is a training-reward view, not verified recovery (strict V2 is 0 by construction because `verifiers.py` requires `outcome.sandboxed`); prompt v1 (the prompt presupposes a fault, see 4d); HF sampling at T=1.0, top_p 0.95; sealed TEST untouched. Sources: `PILOT_REPORT.md`, `CONTROLS_REPORT.md`, `SEEDS_3_5_REPORT.md`, `BASE_EVAL.md`, `BASE_EVAL_EXPLORATORY.md`, `validity/`, `validity2/`.

### 4a. Pilot (seed 0): base -> SFT -> GRPO (`PILOT_REPORT.md`)
Oracle SFT on 64 TRAIN episodes (3 epochs, 48 optimizer steps, 406,731 tokens, loss 1.39 -> 0.050, 242 s, L40S). GRPO: 20 optimizer steps, group 8, 2 prompts/step, max_new 400, lr 2e-5 (not tuned), beta 0, DAPO loss, A100-80GB, ~275-286 s per arm.

| model | valid | V0 | V1 | expl. V2 | mixed-group V2 |
|---|---|---|---|---|---|
| base Qwen3-4B | 48.4% | 13.0% | 0.5% | 0.26% | 2.1% |
| SFT (16 steps) | 97.7% | 32.6% | 2.9% | 2.1% | 14.6% |
| SFT (48 steps) | 99.2% | 49.0% | 16.7% | 15.4% | 56.3% |
| SFT + GRPO, V0 reward | 99.5% | 64.6% | 25.3% | 24.5% | 54.2% |
| SFT + GRPO, V2 reward | 99.0% | 65.4% | 28.6% | 26.3% | 52.1% |

Base strict V2 is 0.0% and one sample reaches exploratory V2 (`BASE_EVAL.md`, `BASE_EVAL_EXPLORATORY.md`). Both GRPO arms improve on SFT; the single-seed V2-vs-V0 gap (26.3 vs 24.5%) is 7 samples of 384 and is noise. B_OK (healthy job) is 0 in both GRPO arms (SFT 4%).

### 4b. Controls (`CONTROLS_REPORT.md`)
- **Compute-matched continued SFT** (20 more oracle-SFT steps, batch 8, lr 2e-5, 160 sequences, same count as the GRPO arms; GRPO additionally spends rollout compute): expl. V2 0.224 vs GRPO 0.245 (V0 arm) and 0.263 (V2 arm). Continued SFT recovers most of the GRPO gain; the GRPO improvement is not distinguishable from more oracle SFT. One run, no replicates.
- **Menu order/id shuffle** (eval-only): expl. V2 drops 1.9 pts (SFT), 4.4 (GRPO-V0), 3.9 (GRPO-V2); GRPO gains survive (0.20-0.22 vs SFT 0.135). Mostly not a position/id shortcut.
- **Menu-only blind** (SFT model, no config/records): V0 0.315, V2 0, repair chosen 0.284 vs chance 0.167.

### 4c. Six paired seeds, V2-arm minus V0-arm (`SEEDS_3_5_REPORT.md`)
Same SFT init, 20 steps, G=8, seed shared within a pair.

| seed | V0 arm expl. V2 | V2 arm expl. V2 | diff (pts) |
|---|---|---|---|
| 0 | 0.245 | 0.263 | +1.8 |
| 1 | 0.214 | 0.266 | +5.2 |
| 2 | 0.195 | 0.255 | +6.0 |
| 3 | 0.159 | 0.260 | +10.2 |
| 4 | 0.148 | 0.234 | +8.6 |
| 5 | 0.193 | 0.154 | -3.9 |

Mean +4.64 pts, sd 5.08, paired t = 2.24 (df 5), two-sided p = 0.075, 95% CI [-0.69, +9.98] pts. On the V0 metric the mean difference is -0.13 pts (p = 0.951). Decision rule fixed in advance (`docs/DESIGN_NOTES.md`, 2026-10-18): CI spans 0 -> **INCONCLUSIVE**; spend on V2-vs-V0 RL for this task and menu stopped. Against the continued-SFT control (a single run treated as a constant, so optimistic): V0 arm -3.2 pts (p = 0.08), V2 arm +1.5 pts (p = 0.44); neither is a clear win over more SFT. The earlier 3-seed figure (+4.3 pts, CI [-1.2, +9.8], p = 0.077; `validity2/v1v2_seeds_out.json`) and the 3-seed episode-bootstrap CI [+1.3, +7.4] (`CONTROLS_REPORT.md`) ignored seed variance and are superseded by the 6-seed paired t.

### 4d. Main design lesson and validity findings
- **V1 == V2 on 97.7% of 3072 DEV rollouts** (`validity2/v1v2_seeds_out.json`; A1 89.4%, A2 99.8%, B_SKIP 99.8%, B_DUP and B_OK 100%; V1 passes where V2 fails on 70 rollouts). Every menu has exactly one repair that recovers the job (`validity/validity_stdout.txt`: one ok candidate per episode, only the `correct` role), so the recovery requirement adds almost nothing beyond label + evidence and V2 collapses to V1. A V0-vs-V2 comparison on this menu design cannot isolate the value of recovery verification. The fix recorded in `docs/DESIGN_NOTES.md` is menus with at least two label-consistent options that complete the run, each distractor violating one V2 invariant, with outcomes from live re-execution.
- **B_OK ~0% is a prompt flaw** (`validity2/bok_diag_stdout.txt`): the prompt presupposes a fault and mentions `NO_FAULT` only in the label list. On healthy B_OK DEV episodes the GRPO-V0 adapter answered `CKPT_SAMPLER_INCONSISTENT` 71/72 times. A neutral prompt v2 is prepared (`modal_train/prompt_v2_cache.py`, `cache_v2prompt/`); **no model has been retrained on it**, so all numbers above use the old prompt.
- The DEV tables come from the `dev_container` path. Table-vs-sandbox agreement has not been shown at scale; only a small Modal Sandbox proof exists (section 4f).

### 4e. Cost (`modal_train/ledger.jsonl`)
Sum of `kind == "actual"` rows: **$8.04 actual** (418 `kind == "actual"` rows, snapshot 2026-10-18, still growing while jobs run): eval $2.99, GRPO $2.15, red-team $2.27 (includes the errored 403 run, isolation and smoke runs; the v1 baseline itself was $0.455), CPU S1 $0.30, SFT $0.24, sandbox proof $0.07, synthetic smoke $0.01. `projected` rows are budget-guard reservations, not spend.

### 4f. Modal Sandbox verification backend (proof only; `results/MODAL_SANDBOX_PROOF.md`)
Task A2, seed 101, world 3, `modal.Sandbox.create(block_network=True, cpu=4, memory=4096)` (gVisor). Network blocked (urllib and DNS fail), loopback Gloo works. The gold repair was ACCEPTED (12 steps, lost_steps=1, `exec_mode=modal_sandbox`), repeated deterministically; two fake repairs were REJECTED on I1 `no_completed_relaunch` (F06 raise-timeout, F11 `world_size:1`). Four verifications, ~187 s, ~$0.07. This proof is one scenario and two cheats; the red-team result is section 4g.

### 4g. Red-team results (v1 baseline, Modal Sandbox)
Source: the committed table `docs/REDTEAM_TABLE.md` (commit `fceb762`), recounted 2026-10-18 with the awk command in `docs/RELEASE_MANIFEST.md`. **The raw v1 JSON was overwritten by a later errored re-run (ProxyError 403 / budget cap) and is not recoverable; these numbers rest on the committed table, not on a re-readable JSON.** Every row ran with `exec_mode = modal_sandbox` (gVisor, no network; invariants recomputed outside the sandbox). 107 cases over five scenario kinds (A1_src, A1_cfg, A2, B_SKIP, B_DUP), $0.455 booked.

| group | result |
|---|---|
| null (no diff) | 0 of 5 accepted (3 execution_failure, 2 invariant) |
| gold repairs | 5 of 5 accepted |
| alternative valid repairs | 9 of 10 accepted; the one false reject is B_DUP alt2 (sampler position recomputed in `restore()`: samples, params, transitions and restores pass; only the checkpoint-format check fails because saved checkpoints store no sampler position). A verifier-strictness finding, not patched in v1 |
| borderline S01 (A2: skip the shard wait) | accepted; a surface patch that removes the injected cause; reported, not counted as a cheat (motivated A2v2) |
| cheats (19 classes F01-F17, F03b, X1, X2) | **0 of 86 accepted**; pooled rule-of-three 95% bound <= 0.035; per-class n is 2 to 5, so per-class bounds are weak (0.6 to 1.5) |
| ...by rejection class, reported separately | **48 by an artifact invariant, 38 by execution_failure** (no relaunch completed) |

Execution failures show the cheats do not work, not that invariants would catch them; invariant coverage is claimed for the 48 only. No claim of completeness over unlisted cheat classes; the audit is by the project's own agents, with no external red team.

Additional evidence (separate, not in the counts above): (a) verifier v1.1 (opt-in, **not active**, not owner-approved), same suite: cheats 0 of 85 verified accepted, gold + alt-gold 15 of 15 (B_DUP alt2 accepted), $0.625 (`docs/results/REDTEAM_TABLE_v11.md`, `docs/VERIFIER_V11.md`); (b) A2v2, a separately versioned scenario with a real producer thread: S01 and R1 rejected by sample accounting, valid producer-side repairs accepted, S02 (derive positions from the sampler) accepted as valid (`docs/A2V2_SCENARIO.md`, `docs/results/A2V2_TABLE_*.md`). Not done: external red team, live sealed-TEST run.

### 4h. Not done
(Sealed TEST: done once, see section 6.) OS-sandboxed or live re-execution of the full fix table; retraining under prompt v2 and distractor menus; the preregistered 20-point confirmatory test (the observed effect is about 4.6 points, well below the minimum meaningful difference, and the study is exploratory); SFT/GRPO with a held-out mechanism split (splits are by seed).

## 5. Known issues in this pipeline
- `test_prompt_is_not_copied_into_modal_train` trips on `modal_train/synth.py` (pre-existing on the branch).
- Cached bundles embed the old menu and key-derived candidate ids: re-run `prepare` after the menu/cap changes and with a fresh `TD_KEY` (`modal_train/README.md`, `docs/audits/leak_validation_97a49d8.md`).
- Eval on Modal uses HF `generate` (T=1.0, top_p 0.95), not the MLX sampler, so numbers are not directly comparable to MLX runs.

## 6. Sealed TEST result (run once 2026-10-18/19; EXPLORATORY)
100 fresh episodes (20 per kind: A1, A2, B_SKIP, B_DUP, B_OK), menu v1, prompt v1, 15 policies, G=4. V0/V1 are scored offline; V2 is a live replay of every unique (episode, chosen repair) pair in the network-less Modal Sandbox (369 pairs, all `exec_mode=modal_sandbox`, 0 errors). Source: `docs/results/TEST_REPORT.md`; independent recount: `docs/results/test/AUDIT_RECOUNT.txt` (`scripts/audit_test_recount.py`).

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

