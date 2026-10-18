# Preregistration: V0-vs-V2 overall and by stratum (config-obvious vs evidence-ambiguous)

Status: written before any V0- or V2-trained model has been evaluated on DEV or TEST. Basis: owner/tech-lead decision
(docs/DESIGN_NOTES.md, m0-bootstrap 3d7e58b): A1-cfg/A2 configuration stays visible because it is real causal evidence;
V0-vs-V2 is reported overall AND on two strata. Verifier, reward and gate semantics are unchanged by this document.

## 1. Stratum assignment RULE (structural, outcome-free)
A scenario is **config-obvious** iff the injected bug edits `job/config.yaml` (equivalently: the repair location is the
config the bundle shows). Otherwise it is **evidence-ambiguous** (the config is clean; the cause must be read from
records/source). The rule is computed from the generator (`jobs/build.py::build`) and uses no model output.

| variant | bug edit | stratum |
|---|---|---|
| A1_cfg (A1, variant 1) | config: one rank's sync mode `reduce_bcast` | config-obvious |
| A2 | config: a rank's shard dir `part-<world>` | config-obvious |
| A1_src (A1, variant 0) | train.py rank-keyed branch | evidence-ambiguous |
| B_SKIP, B_DUP | ckpt.py | evidence-ambiguous |
| B_OK (no fault) | none | evidence-ambiguous (config silent; correct answer is no-fault) |

Expected counts on DEV (seeds 10000-10399, from the generator only): config-obvious 136 (A1_cfg 50, A2 86),
evidence-ambiguous 264 (A1_src 35, B_SKIP 63, B_DUP 81, B_OK 85). Counts vary with seed draws: a stratum is reported
only if it has >= 100 DEV episodes.

## 2. Config-only baseline (TRAIN seeds 20000-20399 only; no DEV/TEST outcome of any model used)
Rule written from config semantics, not fitted: a per-rank sync mode other than `all_reduce` -> A1; a shard dir other
than the rank's own -> A2; config silent -> majority label among config-silent TRAIN scenarios (CKPT_SAMPLER_INCONSISTENT).
Script `docs/prereg/strata_config_baseline.py`, output `strata_config_baseline.json`.

| subset (TRAIN) | n | config-only label accuracy |
|---|---|---|
| config-obvious | 124 | 1.000 |
| evidence-ambiguous | 276 | 0.533 (= share of B_SKIP+B_DUP; A1_src and B_OK are not separable from config) |
| overall | 400 | 0.678 (label majority 0.368) |

Consequence: on the config-obvious stratum label accuracy is not evidence of reading logs; V0 and V2 may both reach it
from the config. The informative contrasts are (a) evidence-ambiguous stratum and (b) recovery-verified success.

## 3. Endpoints
**Primary (single, confirmatory design):** overall live-verified recovery rate (fresh trusted re-execution of the
agent's permitted patch; claimed vs verified logged) on DEV, V2-trained minus V0-trained.
Matched across arms: same base model and LoRA config; same TRAIN seed list and order; same prompts (byte-identical
rendering, one candidate-id key per run set); same sampling (G = 8 rollouts per prompt, temperature 1.0, top_p 0.95,
same max_tokens, thinking off); same optimizer steps, tokens generated and verifier calls (reported, not just steps);
same wall-clock accounting; same task-selection protocol (no adaptive curriculum in the primary comparison). Only the
reward differs (V0 vs V2). Evaluation: greedy or fixed-temperature decoding identical for both arms, k >= 4 epochs
per DEV scenario, intervals clustered by scenario family (A1, A2, B).
**Secondary (exploratory, same arms/inputs):** (S1) live-verified recovery on config-obvious; (S2) on
evidence-ambiguous; (S3) V0-vs-V2 disagreement rate per stratum; (S4) label accuracy and evidence-grounded accuracy per
stratum; V2 components reported separately. Multiple secondaries: no claim from one without the primary.

## 4. Minimum meaningful difference and falsification (the design spec (docs/DESIGN_NOTES.md))
MMD = 15 percentage points on the primary (the effect size section 8 sizes the power analysis for); the power analysis
must confirm or revise this number and is frozen before any run. Outcomes, decided on the 95% CI of the paired
difference (V2 - V0) in live-verified recovery on held-out variants:
- **Falsified:** the CI excludes a positive effect >= MMD.
- **Inconclusive (underpowered):** the CI spans both 0 and MMD.
- **Practically insignificant:** the effect is real (CI excludes 0) but below MMD.
- Supported only if the CI lies above 0 and its lower bound >= MMD, in the primary and in at least the evidence-ambiguous
  stratum; a gain confined to config-obvious is reported as "config-readable", not as evidence learning.
Never silently downgrade to SFT; null results are reported plainly.

## 5. Seeds
One training seed per arm is **exploratory** and is labelled so everywhere; no confirmatory claim without >= 3 paired
seeds x 2 arms (a robustness minimum, not power). TEST (private seed pool) stays sealed and is not used to tune anything here.
Seed-only splits do not support held-out-mechanism claims (docs/splits.md).

## 6. Limits
Two task families; population-level findings are exploratory. Config-visible causes are intentionally kept; see
docs/audits/config_visibility_note.md for the remaining 1:1 value-to-cause mapping and neutral alternatives.
