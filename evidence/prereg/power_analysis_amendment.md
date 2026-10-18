# Amendment to strata_v0_v2.md: power analysis and revised MMD

Status: written and committed BEFORE any V0- or V2-trained model is evaluated on DEV or TEST. It does not edit the frozen
preregistration (f2fda7d, sha256 5ab8a4b1...); section 4 of that file said the power analysis "must confirm or revise" the
provisional 15-point MMD. This amendment revises it. Script: `docs/prereg/power_analysis_sim.py` (seed 20261018, 300
reps per grid point, 600 per null check; `python docs/prereg/power_analysis_sim.py 300`), results:
`docs/prereg/power_analysis_results.json`. Simulation only; no model output is used.

## Model (assumptions, all unmeasured and stated)
DEV = 400 prompts in 6 variant clusters (50/35/86/63/81/85) and 3 families (A1 85, A2 86, B 229). Per-prompt latent
difficulty is shared by both arms (pairing). Logit-scale variance 1, split by ICC rho into cluster and prompt parts;
the treatment effect also varies by cluster with variance rho. Each arm x training-seed has a shift ~ N(0, 0.4^2) (an
assumed training-run noise; must be re-estimated from the first real seeds). G samples per prompt per arm, k=G=4 (8 checked:
no material change). Base rate = V0 verified-recovery rate in {0.1, 0.3, 0.5}; 0.678 is the config-only label accuracy
(a label-accuracy secondary, not a recovery base rate). rho in {0.1, 0.3, 0.5}; seeds S in {1, 3, 5} per arm.
Reject = 95% two-sided CI of the paired difference (V2 - V0) excludes 0, estimate positive.

## Results: minimum detectable difference at 80% power, alpha 0.05 (percentage points)
Base 0.3 (0.1 / 0.5 in text), analysis; "type I" = rejection rate at true difference 0.

| rho | S | variant-cluster CI (6, df5) | family-cluster CI (3, df2) | seed-level paired t (df S-1) |
|---|---|---|---|---|
| 0.1 | 1 | 17 (type I 0.26) | 20 (0.19) | not estimable |
| 0.3 | 1 | 23 (0.17) | 30 (0.12) | not estimable |
| 0.5 | 1 | 25 (0.14) | 34 (0.08) | not estimable |
| 0.3 | 3 | 19 (0.11) | 27 (0.09) | 35 (0.05) |
| 0.5 | 3 | 22 (0.09) | 33 (0.08) | 36 (0.06) |
| 0.3 | 5 | 17 (0.07) | 27 (0.06) | 20 (0.05) |
| 0.5 | 5 | 23 (0.06) | 32 (0.06) | 21 (0.09) |

Across base rates, 5 seeds, seed-level test: 11-13 points at base 0.1, 19-21 at 0.3, 19-21 at 0.5 (rho 0.1-0.5). A naive
prompt-level paired test (ignores clustering and run noise) shows MDD 4-14 points but type I 0.35-0.5: invalid, never used.

## Findings
1. Cluster-robust CIs (variant or family) do not include training-run noise, so with 1 seed they are anti-conservative
   (type I 0.08-0.28 at nominal 0.05). A single seed can never be confirmatory; it stays exploratory (as in section 5).
2. Three seeds with the valid seed-level test (df 2) detect only about 26-36 points; 3 seeds is a robustness check, not power.
3. Five paired seeds with the seed-level test have nominal type I (0.03-0.09) and MDD about 20 points (base >= 0.3) or about
   12 (base 0.1). More DEV prompts do not fix this: the floor is set by seeds and clusters, not by n=400.

## Revised preregistration (replaces the 15-point figure in section 4; everything else in strata_v0_v2.md stands)
- **MMD = 20 percentage points** on the primary endpoint (overall live-verified recovery, V2 - V0). 15 points is detectable
  only if rho <= 0.1 and base <= 0.1 or with more than 5 seeds.
- Confirmatory design: **>= 5 paired seeds x 2 arms**, primary CI = seed-level paired t (df 4) on per-seed overall
  differences; cluster-robust (variant) CI reported alongside. Falsified / inconclusive / practically-insignificant / supported
  rules in section 4 apply with MMD = 20.
- 1 seed or 3 seeds: exploratory, reported with the cluster CI and an explicit note that it excludes run noise.
- Re-estimate run noise (sd of per-seed overall rate within arm) from the first 3 seeds; if it exceeds 0.4 logit the MMD
  is raised, if lower it may be lowered, but only by a new dated amendment file written before the remaining seeds are scored.
- Stratum secondaries (S1-S4) have fewer clusters and are underpowered for any difference below about 30 points; they stay exploratory.
