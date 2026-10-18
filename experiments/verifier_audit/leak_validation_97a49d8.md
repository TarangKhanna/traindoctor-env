# Re-validation after the menu-leak fix (code commit 97a49d8)

> Historical note: this page was written when the Tier A prompt cap was 2,048 tokens. The cap was amended to 3,072 on 2026-10-18 (`evidence/prereg/amendment_token_cap.md`) and `CAP_TOKENS = 3072` in `traindoctor_env/bundle.py` is current. The 2,048 figures below are historical.

Follows `leak_validation_796ba23.md` (kept unchanged: it is the failure record). Scripts and JSON next to this file:
`leak_validation_97a49d8.py/.json` (the 796ba23 validator re-run unchanged in logic), `_ablation.py` (copied, not re-run).
400 DEV seeds 10000..10399, 5-fold CV across seeds, Wilson 95% CIs, `TD_KEY` set (dev key). CPU only; no Gloo/MLX/modal.
No verifier, reward or strict-gate code changed.

## Changes (traindoctor_env/bundle.py)
1. Every menu item's context now comes from the CLEAN job text, rendered the same way. The no-change item takes its
   context lines from the clean config (`Candidate.clean_cfg`); the exclude item uses the same neutral two-form display as
   the location items (`_shown_parts("exclude", ...)`). Previously both used the buggy text, so `reduce_bcast` appeared in
   those two items' context in 50/50 A1-per-rank seeds and nowhere else.
2. Candidate ids are `"R" + sha256(salt_for(seed) + "|menu-id|" + role)[:5]`, keyed by the TD_KEY-derived per-scenario
   secret (`jobs.dataserver.salt_for`). Deterministic for a given key and seed; changing the key changes every id.
   Evidence (record) ids are untouched. Anything that stores candidate ids (fix-outcome tables, SFT data, DEV rebaselines)
   is valid only for the key it was generated with; the DEV rebaseline listed as required in 796ba23 stays required.

Distinction kept: the bundle config alone reveals A1-cfg (`reduce_bcast`) and A2 (`part-<world>`). That is evidence the
agent is meant to read, not a menu leak; it appears below only as the "menu + config" control, which is expected to be
above chance. The rule tested is about the menu alone.

## Results (menu-only, out-of-fold accuracy [Wilson CI]; majority = 0.360 for repair and label)
| target | majority | stumps | NB | LR | best lookup |
|---|---|---|---|---|---|
| repair slot (796ba23: stumps 0.495) | 0.360 [0.315, 0.408] | 0.333 [0.288, 0.380] | 0.237 | 0.273 | 0.345 |
| label (796ba23: stumps 0.485) | 0.360 | 0.300 [0.257, 0.347] | 0.258 | 0.223 | 0.343 |
| kind (5 classes; majority 0.215) | 0.158 | 0.170 [0.136, 0.210] | 0.155 | 0.188 | 0.220 |

With the learned confusion remap, no predictor exceeds 0.372 on repair/label. Item-level scorer (repair): stumps 0.358
[0.312, 0.406], NB 0.36, LR 0.25. Counterfactual groups (200 groups x 6 variants, train natural 10200-10399, chance 0.333):
repair stumps 0.273 [0.248, 0.298], NB 0.256, LR 0.165; label stumps 0.266, NB 0.258, LR 0.207. All at or below chance.
Control (menu + padded config, no records): stumps 0.71 for both repair and label (0.36 + 0.125 + 0.215 as before): the
config evidence, expected.
Id probe: seed recovered from the old public formula 0/100 (was 100/100); regression test uses 50 menus.

## Regression tests (tests/test_menu_leak.py, audit/menu_leak.py)
- `test_boosted_stumps_on_menu_only_features_do_not_beat_majority`: rule = no menu-only predictor beats majority by more
  than z=3 standard errors (0.072 at n=400) on held-out seeds, repair and label. Checked that it can
  fail: on the pre-fix code at 400 seeds it gives repair 0.4475 (+0.0875 > margin 0.072) and label 0.46 (+0.100).
- `test_candidate_ids_are_not_a_public_function_of_seed_and_role`: old public formula recovers 0 menus; a different key
  gives disjoint ids; same key and seed gives the same id.
- Passing: test_menu_leak, test_token_cap (non-executed), test_bundle_leak, test_scenarios, test_tiera (non-executed): 11
  passed in the first batch run; menu_leak 5/5 after adding the two tests.

## NOT RUN
- Gloo / sandbox-exec tests (e.g. `test_token_cap` executed-bundle cases fail here: sandbox-exec unavailable), so: every
  decoy and repair re-execution outcome, A1 `if False:` and B ckpt single-site edits under re-execution, real-record
  bundle-evidence oracle (the synthetic partial from 796ba23 was not re-run), executed true-token counts <= 2048 with the
  new exclude display, B_SKIP vs B_DUP separation.
- The ablation script was copied but not re-run (its premise, buggy-context items, no longer exists).
- Stumps were not run under a label-permutation null. Validation is by the same author as the fix; an independent re-run is
  still advisable.
