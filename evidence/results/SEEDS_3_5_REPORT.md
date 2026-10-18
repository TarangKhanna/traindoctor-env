<!-- provenance: copied unmodified below this line from modal_train/results/SEEDS_3_5_REPORT.md (gitignored original) on 2026-10-18; exec_mode dev_container-EXPLORATORY; see docs/RELEASE_MANIFEST.md -->
# Seeds 3-5 and 6-seed paired report (EXPLORATORY dev_container, not verified recovery)

48 DEV eps x G=8 = 384 samples per run; sft1 full init, 20 optimizer steps, G=8, lr 2e-5, seed shared within each V0/V2 pair; sealed TEST untouched. All 12 evals + csft re-scored with `analyze_exploratory.py` (source *_eval.json -> *_EXPLORATORY.json), so seed 0-2 values are from the same scorer as 3-5. Seed 0 = pilot (eg_v0/eg_v2).

## Per-run (exploratory V2 / V0 / V1)
| seed | arm | expl V2 | V0 | V1 | valid |
|---|---|---|---|---|---|
| 0 | V0 | 0.245 | 0.646 | 0.253 | 0.995 |
| 0 | V2 | 0.263 | 0.654 | 0.286 | 0.990 |
| 1 | V0 | 0.214 | 0.724 | 0.245 | 0.987 |
| 1 | V2 | 0.266 | 0.703 | 0.297 | 0.984 |
| 2 | V0 | 0.195 | 0.607 | 0.214 | 0.992 |
| 2 | V2 | 0.255 | 0.648 | 0.281 | 0.997 |
| 3 | V0 | 0.159 | 0.586 | 0.167 | 0.995 |
| 3 | V2 | 0.260 | 0.591 | 0.271 | 0.990 |
| 4 | V0 | 0.148 | 0.526 | 0.167 | 0.997 |
| 4 | V2 | 0.234 | 0.573 | 0.258 | 0.984 |
| 5 | V0 | 0.193 | 0.633 | 0.203 | 0.990 |
| 5 | V2 | 0.154 | 0.544 | 0.161 | 1.000 |

## 6-seed paired table (V2 arm minus V0 arm)
| seed | V0 arm | V2 arm | diff (pt) | V0-metric diff (pt) |
|---|---|---|---|---|
| 0 | 0.245 | 0.263 | +1.8 | +0.8 |
| 1 | 0.214 | 0.266 | +5.2 | -2.1 |
| 2 | 0.195 | 0.255 | +6.0 | +4.2 |
| 3 | 0.159 | 0.260 | +10.2 | +0.5 |
| 4 | 0.148 | 0.234 | +8.6 | +4.7 |
| 5 | 0.193 | 0.154 | -3.9 | -8.9 |

expl V2: mean diff +4.64 pt, sd 5.08, t=2.24 (df=5), two-sided p=0.075, 95% CI [-0.69, +9.98] pt (paired t over seeds; n=6, training-seed variation only).

V0 metric: mean diff -0.13 pt, sd 4.96, t=-0.06 (df=5), two-sided p=0.951, 95% CI [-5.33, +5.07] pt (paired t over seeds; n=6, training-seed variation only).

## GRPO arms vs continued SFT (csft expl V2 0.224, single run, no seed replicates)
Test = one-sample t of the 6 seed-level arm scores against the csft score (treats csft as a fixed constant; ignores its own seed/eval noise, so optimistic).
| arm | mean | sd | mean - csft (pt) | t (df=5) | p | 95% CI (pt) |
|---|---|---|---|---|---|---|
| V0 arm | 0.192 | 3.5 | -3.2 | -2.19 | 0.080 | [-6.9, +0.5] |
| V2 arm | 0.239 | 4.3 | +1.5 | 0.84 | 0.440 | [-3.1, +6.0] |

## GRPO training (seeds 3-5)
| run | optimizer steps | nonzero-grad logged entries | adapter sha256 (12) |
|---|---|---|---|
| g_v0_s3 | 20 | 10/10 | cdaffe81afd0 |
| g_v2_s3 | 20 | 10/10 | 0bafce47da9c |
| g_v0_s4 | 20 | 10/10 | 803d9a3502e5 |
| g_v2_s4 | 20 | 9/10 | 8e29cd731b2a |
| g_v0_s5 | 20 | 9/10 | 4a81dc049052 |
| g_v2_s5 | 20 | 10/10 | bf07cb5eb2ee |

## Spend
Seeds 3-5 lane (6 GRPO + 6 evals) $1.94; ledger actual total $5.69 of $30 cap. Note: the first eval launch (19:11) died on a sandbox network block (api.modal2.com), no compute billed; relaunched with host allowed.
