<!-- provenance: copied unmodified below this line from modal_train/results/PILOT_REPORT.md (gitignored original) on 2026-10-18; exec_mode dev_container-EXPLORATORY; see docs/RELEASE_MANIFEST.md -->
# Pilot report (2026-10-18): Qwen3-4B oracle SFT + 20-step GRPO, V0 vs V2 arms
ALL DEV OUTCOMES ARE EXPLORATORY (dev_container tables, dev_episode sandboxed=True re-score; strict V2 is 0 by construction). Single seed (0), 48 DEV episodes x G=8 = 384 samples; no CI/significance claim. Sealed TEST untouched.
Data: modal_train/cache/PROVENANCE.md (source tdrun td_dev_prepare-20261018T135749-5051; sha256 listed there; train 64 eps/384 cand/64 ok/13 fault_not_injected; dev 48/288/48/10).

## Runs (all L40S $1.95/h except GRPO A100-80GB $2.5/h; run json in results/*_sft.json, *_grpo.json, *_eval.json)
- sft1: TRAIN oracle SFT, LoRA r16, lr 1e-4, batch 4 accum 1, 3 epochs = 48 optimizer steps, 406,731 tokens; loss 1.39 (step1) -> 0.076 (17) -> 0.050 (48); 242 s, $0.131, peak 13 GB.
  full adapter (48 steps) sha256 8d96bc11...ebe02 (volume td-results:/sft1/adapter); light (16 steps) sha256 d9dab38e...6199 (/sft1/adapter_light).
  `TD_GPU=L40S modal_train/run.sh 2400 run_sft '--run-id sft1 --epochs 3 --batch 4 --accum 1 --save-at 16'`
- Evals (`run_eval --run-id X --adapter-run R [--adapter-name adapter_light] --g 8 --batch 2`, ~270 s, ~$0.15 each).
- g_v0 / g_v2: `TD_GPU=A100-80GB run_grpo --run-id g_vX --arm VX --sft-run sft1 --steps 20 --lr 2e-5` (group 8, 2 prompts/step, max_new 400, beta 0, dapo loss, same seed/task order). ~275-286 s each, $0.19-0.20, peak 20 GB. Run concurrently on separate GPUs. LR 2e-5 was my choice (not tuned).

## Exploratory DEV results (G=8, n=384)
| model | valid | V0 | V1 | expl. V2 | mixed-group V0 | mixed-group V2 | mean tok |
|---|---|---|---|---|---|---|---|
| base | 48.4% | 13.0% | 0.5% | 0.26% | 8.3% | 2.1% | 98 |
| SFT light (16 st) | 97.7% | 32.6% | 2.9% | 2.1% | 87.5% | 14.6% | 50 |
| SFT full (48 st) | 99.2% | 49.0% | 16.7% | 15.4% | 91.7% | 56.3% | 50 |
| GRPO V0 arm | 99.5% | 64.6% | 25.3% | 24.5% | 37.5% | 54.2% | 51 |
| GRPO V2 arm | 99.0% | 65.4% | 28.6% | 26.3% | 58.3% | 52.1% | 51 |
Per-kind expl. V2 (A1/A2/B_SKIP/B_DUP/B_OK): SFT full .17/.21/.19/.14/.04; V0 arm .19/.36/.31/.35/0; V2 arm .24/.55/.29/.21/0. Failure reasons (V0 arm / V2 arm): wrong_label 134/129, wrong_repair_or_evidence 154/150, bad_schema 2/4. Full per-arm JSON/MD: *_EXPLORATORY.*.

## Gates
valid 99.2% >= 90 pass; mixed expl-V2 groups 56% >= 25% pass (light checkpoint 14.6% would fail); SFT expl V2 15.4% < 90% pass (headroom). Full SFT used for GRPO.

## GRPO diagnostics (log has 10 reward entries for 20 optimizer steps; TRL logs per generation batch)
- optimizer steps 20 each; nonzero grad_norm in all logged entries except V2 entry 2 (grad 0, all-zero reward group); zero-variance fraction per logged entry 0-0.5 (V0) and 0-1.0 (V2).
- Reward mean per logged entry: V0 [.25 .38 .75 .88 .81 .50 .94 .94 .25 .44]; V2 [.25 0 .38 .44 .38 .06 .62 .56 .50 .06] (noisy: 2 prompts/step, different rewards per arm, not comparable across arms).
- Adapter delta vs SFT (L2): V0 0.380, V2 0.395 (1.0-1.1% of adapter norm). adapter sha256: V0 dc91a007...5727, V2 24ef036d...6a2ba01. Seconds: rollout ~138/123, backward+other ~117/124.

## Caveats
Both arms improved V0/V1/V2 similarly over SFT; V2-vs-V0 expl. V2 difference (26.3 vs 24.5%) is 7 samples of 384 from one seed -- not distinguishable from noise, no claim. B_OK is 0 in both arms (SFT 4%). Evaluation episodes were scored with the DEV training-reward view; DEV tables are UNVERIFIED. Eval uses HF sampling T=1.0 top_p 0.95. Logs of concurrent launches share names if started in the same second (run.sh); results are from volume JSONs.

## Spend
Ledger total $1.69 (was $0.59 before; pilot $1.11 of the $6 limit). `modal billing summary` (results/billing_after_pilot.txt): metered $2.81 total workspace incl. volumes $0.87, billed $0.00 (credits). Not authorized beyond $6 pending Opus approval.
