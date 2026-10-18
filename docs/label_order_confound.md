# Label-order confound (experimental flag `--shuffle-labels` in rl/model_screen.py; production prompt/fixtures unchanged)
Qwen3-4B-Instruct-2507, 8 DEV episodes x G=4, Labels line permuted per episode (seeded), strict gate.
- Unshuffled: 30/30 valid answers = COLLECTIVE_DIVERGENCE (V0 .25, V2 .125, no spread).
- Shuffled: 30/32 valid; COLLECTIVE_DIVERGENCE 20, RANK_STALLED_BEFORE_COLLECTIVE 10, never CKPT_SAMPLER_INCONSISTENT or NO_FAULT. V0 .094, V2 0, 1 group with V0 spread, 0 with V2 spread.
Reading: list position explains part of the bias (RANK_STALLED appears only when ordering changes) but the collapse does not disappear; the model still ignores the records. No fixture change warranted by this test.
Finding: real prompt lengths are 1.9k-2.6k tokens while CAP_TOKENS=2048 uses a 4-chars-per-token estimate (bundle.py:105-111); cap not changed.

## Controls (Qwen3-4B-Instruct-2507, same 8 DEV episodes x G=4, labels permuted per episode, strict gate; 32 completions each)
| prompt | valid | valid label counts | V0 mean | V2 | repair ok (of valid) |
|---|---|---|---|---|---|
| full, unshuffled | 30 | COLL 30 | .25 | .125 | 4 |
| full, labels permuted | 30 | COLL 20, RANK 10 | .094 | 0 | n/a |
| menu_only (no config/records) | 32 | RANK 18, CKPT 11, COLL 3 | .531 | 0 | 2 |
| evidence_only (repair contents withheld) | 23 | COLL 12, RANK 11 | .219 | 0 | 6 |
| neutral (labels + opaque repair ids only) | 32 | COLL 14, RANK 8, CKPT 6, NO_FAULT 4 | .062 | 0 | 7 |
True label frequency in these episodes: CKPT 3, COLL 2, RANK 2, NO_FAULT 1 (x4 each).
Reading: with no evidence at all the model spreads over all four labels (no inherent COLL prior; V0 .06), so the full-prompt collapse is NOT a pure label prior. Menu-only reaches V0 .53, consistent with the audit-lane finding that menu content leaks label and repair; full prompt (2-2.8k tokens) does worse than the 1k-token menu-only prompt. No control shows evidence being used (V1 = 0 everywhere). Cause of the full-prompt collapse: unresolved (long-context/instruction-following vs evidence use not separated). Menu leak flagged for M2; production menu unchanged. Gemma: inconclusive.
