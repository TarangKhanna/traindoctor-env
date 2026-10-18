# Amendment: Tier A token cap 2,048 -> 3,072

Date: 2026-10-18. Applies to the frozen `docs/prereg/strata_v0_v2.md` (not edited).

The Tier A prompt cap (`traindoctor_env/bundle.py:CAP_TOKENS`) changed from 2,048 to 3,072 true Qwen3 tokens after the
preregistration and before any outcome evaluation (no V0/V2 arm has produced a result; the only data are the Modal
readiness run's `prepare` abort and the leak audits).

Why: the evidence certificate is always retained (the design spec (docs/DESIGN_NOTES.md)) and for Task B at 3-4 ranks it alone needs 2,121-2,432
estimated tokens (Modal readiness run), so 2,048 made `prepare` abort with `BundleTooLarge` on 5 of 12 DEV seeds. One cap
for all variants, not per variant, so length does not mark the variant.

What did not change: strata definitions, the verifiers V0/V1/V2, the reward, the strict gate, the split by mechanism, the
power analysis. What did change as a consequence: with a higher cap, label-independent noise trimming keeps more
non-certificate records; the leak validations and mutation checks were re-run at the new cap (see
`docs/audits/token_cap_3072.md`).
