# Config visibility, label-encoding check, and what is NOT RUN

> Historical note: this page was written when the Tier A prompt cap was 2,048 tokens. The cap was amended to 3,072 on 2026-10-18 (`docs/prereg/amendment_token_cap.md`) and `CAP_TOKENS = 3072` in `traindoctor_env/bundle.py` is current. The 2,048 figures below are historical.

Code state: modal-cuda on 15bfd41. Tests: tests/test_config_swap.py (pure Python, 6 passed, 0.11 s).

## Findings
- Config key sets are identical for all six variants; no key exists for one cause only. No key or string value contains a
  label, kind or fault word (tests R2, R3).
- Counterfactual swaps (tests R1): the config-visible cause (sync mode deviating from all_reduce; shard dir not the rank's
  own) follows the config text, not the scenario label; the reverse edit removes it.
- Remaining 1:1 values, listed not changed: (1) `sync.per_rank.<r>: reduce_bcast` appears only in A1_cfg, because every
  decoy per_rank entry is `all_reduce`. (2) `data.shard_dirs.<r>: part-<world>` appears only in A2, because every other
  entry is `part-<own rank>`. Both are implemented job behaviour (template train.py handles `reduce_bcast`; data.py
  reads shard_dirs), i.e. real mechanism, which the owner decision keeps visible. But "any deviation from default" is a
  marker an agent or model can learn without reading records.
- Neutral alternatives (not applied; they would change scenarios, so need a decision and a rebaseline): make decoys draw
  from the same value sets, e.g. a benign second sync mode (an implemented, equivalent `all_reduce` variant) on some
  decoy ranks in every kind, and a benign alias shard dir (a full copy, `part-<r>b`) on some decoy ranks, so a deviation
  alone no longer implies the cause. Strata reporting (strata_v0_v2.md) is the mitigation until then.
- Tests are rule-based; not mutation-checked beyond the one assertion fixed while writing them.

## NOT RUN (needs the trusted launcher / Gloo / sandbox-exec / model)
- Any live re-execution: decoy and repair outcomes, A1 source and B ckpt edits under re-execution, B_SKIP vs B_DUP separation.
- Executed true-token counts <= 2048 for bundles; real-record bundle-evidence oracle.
- Any model evaluation (V0/V2 training or DEV/TEST scoring), power analysis, MMD confirmation.
- Fix-outcome tables and DEV rebaseline for the key-derived candidate ids.
- Swap tests on the rendered bundle (needs executed snapshots); only config text was swapped.
- Independent re-run of the menu validators by a different author.
