# Independent validation of 796ba23 (menu leak closed?)

Validator: separate agent, CPU only (no Gloo, no MLX, no modal). 400 DEV seeds 10000..10399, 5-fold CV across seeds,
Wilson 95% CIs. Scripts and JSON next to this file: `leak_validation_796ba23.py/.json` (Parts 1, 3, id probe),
`leak_validation_796ba23_ablation.py/.json` (source of the residual signal), `bundle_evidence_oracle_796ba23.py/.json`
(Part 2, partial). Run with `PYTHONPATH=.:docs/audits`, `HF_HUB_OFFLINE=1`. No reward/verifier/gate code was touched.

## 1. Alternative predictors on menu-only features: FAIL (narrow, one cause)
Targets: repair slot (6 classes; majority 0.360 = checkpoint slot), label (4 classes; majority 0.360), kind (5).
Repair slot, out-of-fold accuracy (remap = confusion remap learned on train folds, applied to held-out; optimistic, inner
predictions were cross-fit on the same folds):

| predictor | acc [CI] | remap |
|---|---|---|
| majority | 0.360 [0.315, 0.408] | |
| lookup (variant set / unmasked numbers / slot order) | 0.333 / 0.328 / 0.333 | 0.347 / 0.362 / 0.335 |
| logistic regression (uni/bi/char-4, ids, pos, len) | 0.260 | 0.263 |
| naive Bayes | 0.335 [0.291, 0.383] | 0.445 [0.397, 0.494] |
| boosted stumps | **0.495 [0.446, 0.544]** | 0.507 [0.459, 0.556] |

Label: stumps 0.485 [0.436, 0.534] vs 0.360; others 0.305-0.375. Item-level scorers (NB/LR/stumps): 0.36/0.285/0.36.
Permutation null (shuffled train labels, true held-out labels; NB 100 perms, LR 20): repair NB null mean 0.255, p95 0.295,
max 0.315; observed 0.335 (above every permutation, but below majority). Stumps were not run under permutation.

Cause, isolated by ablation: the no-change and exclude-rank items draw their context lines from the BUGGY job text, the four
location items from the CLEAN text. In A1 per-rank variant (50/400 seeds) `reduce_bcast` appears in those two items' context
lines (50/50, 0 false positives), so the menu alone identifies the per-rank repair. Deleting the two items drops stumps to
0.350 (repair) / 0.350 (label), NB 0.215 / 0.255, i.e. chance. Not menu-only-new information relative to the bundle (the
config is shown in the bundle with the same `reduce_bcast`), but it contradicts "all menu-only features at or below chance".
The existing test missed it because its predictors are lookup/SGD-LR on masked n-grams; boosted stumps on unmasked tokens find it.

Held-out counterfactual groups (same seed/world/job_seed, six variants A1src, A1cfg, A2, B_SKIP, B_DUP, B_OK; 200 groups,
1200 menus; train on natural seeds 10200-10399): repair stumps 0.473 [0.444, 0.501], NB 0.326, LR 0.167, majority 0.333;
same leak. B_SKIP and B_DUP menus are identical by construction (same label, same slot).

Config-included (menu + padded config, still no records): label stumps 0.703 [0.656, 0.745], repair 0.698 vs 0.360. This is
the A1-cfg (`reduce_bcast`) and A2 (`part-<world>`) bug being visible in the config (0.36 + 0.125 + 0.215 = 0.70). Validity
note, not a menu leak: for those two variants the records are not needed to diagnose the incident.

## 2. Bundle-evidence-only oracle: NOT RUN on real records; partial result
Real records need executed snapshots (Gloo). Partial: certificate facts synthesized in the documented record format, then a
chooser that sees only record text, padded config and the rendered menu text (no roles/slots). 400 seeds, accuracy per
variant: A1-src 35/35, A1-cfg 50/50, A2 86/86, B_SKIP 63/63, B_DUP 81/81, B_OK 85/85 (all 1.0, Wilson lower bounds
0.93-0.96). This shows the displayed menu is sufficient given the certificate facts; it does not show the facts are present
in the real 2048-token bundle. Not decided: B_SKIP vs B_DUP (same slot, both pick the ckpt item).

## 3. Edit fidelity (shown vs executed): PASS with two mismatches, neither exploitable
- A1 source repair: shown as the one-line conditional pair; executed `if False:` replacing the `if rank == R and step >= S...`
  line (400-seed count: 35/35). Equivalent behavior, different text.
- Checkpoint item: shows both sites (sampler write and restore) for every variant; executed edit changes exactly one
  (B_SKIP: restore site; B_DUP: write site; the decoy: write site). Counts 35+50+86+81+85+63 over all roles.
- Config/sync decoys: displayed forms match executed text exactly (all seeds).
Wrong-reason risk: the ckpt slot is correct for both B kinds and a decoy otherwise, so the display of two sites does not make
a wrong pick right; A1's `if False:` has no distinguishing display. No mismatch found that rewards a wrong choice.

## Extra finding: candidate ids are a public hash of (seed, role) - low/medium
`id = "R" + sha256(f"{seed}:menu:{role}")[:5]`. A program that knows the seed range recovers the seed from two ids and then
the correct id: 100/100 menus (DEV range). An LLM cannot compute this, a code-executing agent could. Salt with the private key.

## 4. Still requires Gloo / trusted launcher (NOT RUN)
- Every decoy fails by re-execution (esp. per-rank `reduce_bcast` on B_OK, sync decoy on all kinds, ckpt decoy on A kinds).
- A1 `if False:` repair passes re-execution; B_SKIP/B_DUP ckpt single-site edits pass.
- Real records: certificate present in every executed bundle, no BundleTooLarge, executed true Qwen3 token counts <= 2048.
- Real-record bundle-evidence oracle (Part 2) for all variants; B_SKIP vs B_DUP separation.
- DEV rebaseline / SFT regeneration (still required per the original doc).

## Verdict
1 FAIL (per-rank variant leaks through buggy-text context of the no-change/exclude items; all else at chance)
2 NOT RUN on real records (partial synthetic 6/6 variants at 1.0)   3 PASS (two benign mismatches)   4 NOT RUN.
Suggested fix (not applied): draw the no-change/exclude contexts from the same clean text as the location items, or show the
bundle config in clean-vs-buggy neutral form; add boosted stumps and the label-permutation null to `audit/menu_leak.py`.
