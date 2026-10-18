# Tier A cap 2,048 -> 3,072: offline validation (2026-10-18)

No Gloo here, so bundles are built from records synthesized in the executed format (`tests/synth_snapshot.py`,
`scripts/token_cap_sweep.py`). Calibration: the Task B certificate-only floor at 4 ranks reproduces the Modal readiness
run (synthetic 2121-2433 est. vs measured 2121-2432). Task A synthesizes ~1700 vs measured 2025-2041 (real A has more
records), so A is not exact; both are far under 3,072. Real executed counts remain NOT RUN.

400 DEV seeds (10000..10399), pessimistic records, true Qwen3 tokens of the full prompt: BundleTooLarge 0/400,
max 2449, median 2153, p95 2431 (A1/A2 ~1690-1720; B_* 3 ranks max 2369, 4 ranks max 2449). Same run at cap 2048 raised
BundleTooLarge on 95/400 (all Task B at 4 ranks). Within a world size the B variants have equal length (B_OK included),
so length does not separate NO_FAULT from CKPT_SAMPLER_INCONSISTENT.

Leak checks at the new cap: menu/config-only predictors do not read records, so the cap cannot change them;
`leak_validation_97a49d8.py` re-run, output byte-identical to the committed JSON. Menu-only (5-fold, 400 seeds): label
best 0.343 vs majority 0.36, kind5 best 0.22 vs 0.215; menu+config stumps reach 0.71 (label), known and unchanged
(config is part of the bundle; docs/audits/config_visibility_note.md). `scripts/mutation_check_leak.py`: 17/17 mutants
detected, undetected [], JSON unchanged. `pytest -m "not sweep and not executed"`: 98 passed, 8 failed, 8 skipped; the 8
are Gloo/sandbox-exec (6 executed token_cap cases) and Metal (2 test_first_update) unavailability, as before.
