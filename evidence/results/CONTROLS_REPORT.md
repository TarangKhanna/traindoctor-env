<!-- provenance: copied unmodified below this line from modal_train/results/CONTROLS_REPORT.md (gitignored original) on 2026-10-18; exec_mode dev_container-EXPLORATORY; see docs/RELEASE_MANIFEST.md -->
# Controls report (2026-10-18): continued-SFT, menu-shuffle/blind, paired seeds
ALL EXPLORATORY (dev_container tables, sandboxed re-score; DEV = tuning set; sealed TEST untouched). 48 DEV eps x G=8 = 384 samples per row. Source: `PYTHONPATH=. python -m modal_train.results.analyze_controls` -> CONTROLS_SUMMARY.json (per-run raw: *_eval.json).

## (c) Continued-SFT control (run csft, eval ecsft)
From sft1 full adapter (sha 8d96bc11; csft adapter sha 84f379d2f80d...), oracle SFT, 20 optimizer steps, batch 8 (accum 1), lr 2e-5, fresh optimizer. Matching: GRPO arms do 20 optimizer steps of 8 sequences = 160 sequences (~340k tokens incl. prompts); csft = 20 x 8 = 160 sequences (2.5 passes over 64 TRAIN eps, ~339k tokens). Not matched: GRPO also spends rollout compute. Loss 0.051 -> 0.014 (already near fit).
| model | valid | V0 | V1 | expl V2 | oracle repair chosen |
|---|---|---|---|---|---|
| SFT full (sft1) | .992 | .490 | .167 | .154 | .484 |
| + 20 cont. SFT | .987 | .667 | .255 | .224 | .599 |
| GRPO V0 (s0) | .995 | .646 | .253 | .245 | .635 |
| GRPO V2 (s0) | .990 | .654 | .286 | .263 | .615 |
Continued SFT recovers most of the GRPO gain (V2 .224 vs .245/.263): the pilot GRPO improvement is not distinguishable from "more oracle SFT steps".

## (b) Menu-shuffle (menu order + menu ids permuted/remapped; eval-only, `core.variant_episode`, no core env change; oracle answer scores 48/48 under each variant) and blind baseline
| model | unshuffled V0/V2 | shuffled V0/V2 | drop V2 |
|---|---|---|---|
| SFT full | .490/.154 | .508/.135 | -1.9 pt |
| GRPO V0 | .646/.245 | .635/.201 | -4.4 pt |
| GRPO V2 | .654/.263 | .635/.224 | -3.9 pt |
Menu-only blind (SFT full, no config/records): valid .799, V0 .315, V2 0 (cannot cite records), oracle repair chosen .284 vs chance 1/6=.167; the label is above 4-class chance (.25) only slightly. Order/id shuffle costs GRPO adapters ~4 pt V2 (and V0 ~1 pt): small, mostly not a position/id shortcut; GRPO gains survive (.20-.22 vs SFT .135).

## (d) Paired seeds (same sft1 init, 20 steps, G=8, 2 prompts/step, lr 2e-5, seed shared within a V0/V2 pair)
| seed | V0 arm expl V2 | V2 arm expl V2 | diff | V0 arm V0 | V2 arm V0 |
|---|---|---|---|---|---|
| 0 (pilot) | .245 | .263 | +1.8 | .646 | .654 |
| 1 | .214 | .266 | +5.2 | .724 | .703 |
| 2 | .195 | .255 | +6.0 | .607 | .648 |
Mean V2-minus-V0 = +4.3 pt (V2 arm .261 vs V0 arm .218); 95% bootstrap CI over the 48 episodes (cluster = episode, seed-averaged) [+1.3, +7.4]. Mean < 5 pt but the CI excludes 0, so the stop-rule's "CI contains 0" clause is not met; not extended anyway. Caveat: CI ignores training-seed variance (n=3 seeds; all three diffs positive but seed 0 small); a V0-metric gap shows no pattern (V0 arm higher V0 in seed 1). Exploratory only, no claim.

## Runs (ids = results/<id>_<fn>.json)
csft (sft, L40S 176 s, 20 steps) adapter sha 84f379d2...; ecsft eval. g_v0_s1/g_v2_s1/g_v0_s2/g_v2_s2 (grpo A100-80GB, 20 optimizer steps each, nonzero-grad logged entries 9/10/9/10 of 10, 222-237 s); adapter shas 0c443c1f, 518bd513, a57d14bb, 36003c25 (full in *_grpo.json); evals eg_v{0,2}_s{1,2}, esh_{sft1,g_v0,g_v2}, emo_sft1. Code: modal_train/{core,trainlib,app}.py (seed, init-run, variant options), analyze_controls.py.
## Spend
Ledger $3.75 total (was $1.69): lane spend $2.06 (< $4).
