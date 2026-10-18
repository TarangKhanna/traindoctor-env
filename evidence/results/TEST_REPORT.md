# Sealed TEST report (run once, 2026-10-18/19; EXPLORATORY)

Protocol: `docs/prereg/test_protocol.md` (frozen in commit e2e2560 before any TEST episode was built). Analysis script `scripts/analyze_test.py`, unit tests `tests/test_analyze_test.py` (9 pass).
Seed list: untracked `modal_train/test_seeds.local.json`, sha256 `52fb377289876036db6bc6766af5b3bcd0c67a7faeb8bd99b6a74f306aa4146d`.
Data: 100 episodes, 20 per kind, menu v1 + prompt v1, built on Modal CPU with one fixed TD_KEY (untracked). 15 policies x 100 episodes x G=4 = 6000 rollouts (Qwen3-4B, L40S, T=1.0, top_p 0.95, max_new 400, seed 0, batch 2).
V2 = live replay in the network-less Modal Sandbox of every unique (episode, chosen repair) pair: 369 pairs, all `exec_mode=modal_sandbox`, `sandboxed=True`, 0 replay errors or timeouts, 2 pairs flagged fault_not_injected. Raw: `docs/results/test/` (`test_scored.json`, `test_replay_r0..r4.json`, `test_*_eval.json`, `TEST_REPORT_analysis.json`).

## Script output (`PYTHONPATH=. python scripts/analyze_test.py modal_train/results/test_scored.json modal_train/results/TEST_REPORT_analysis`)
# Sealed TEST results (EXPLORATORY: two task families; no claim about unseen mechanisms)

Episodes: 100 ({'A1': 20, 'A2': 20, 'B_SKIP': 20, 'B_DUP': 20, 'B_OK': 20}); G=4 rollouts per episode per policy. V2 = live Modal Sandbox replay (exec_mode=modal_sandbox).

| policy | n | valid | V0 label | V1 evid | V2 verified | V2 (4 fault kinds) | claimed | claimed, not verified | replay errors | V2 A1 | V2 A2 | V2 B_SKIP | V2 B_DUP | V2 B_OK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 400 | 51.7 | 15.8 | 4.8 | 4.0 | 5.0 | 199 | 183 | 0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sft | 400 | 99.8 | 47.2 | 17.5 | 15.5 | 17.2 | 399 | 337 | 0 | 16.2 | 12.5 | 23.8 | 16.2 | 8.8 |
| csft | 400 | 100.0 | 67.5 | 30.0 | 25.2 | 29.7 | 400 | 299 | 0 | 37.5 | 21.2 | 36.2 | 23.8 | 7.5 |
| v0_s0 | 400 | 99.8 | 61.8 | 23.2 | 21.5 | 26.9 | 399 | 313 | 0 | 15.0 | 27.5 | 33.8 | 31.2 | 0.0 |
| v0_s1 | 400 | 100.0 | 69.8 | 26.5 | 22.0 | 27.5 | 400 | 312 | 0 | 41.2 | 21.2 | 28.7 | 18.8 | 0.0 |
| v0_s2 | 400 | 99.8 | 57.2 | 24.2 | 20.8 | 25.6 | 399 | 316 | 0 | 46.2 | 12.5 | 18.8 | 25.0 | 1.2 |
| v0_s3 | 400 | 100.0 | 60.8 | 18.8 | 16.5 | 20.3 | 400 | 334 | 0 | 22.5 | 16.2 | 22.5 | 20.0 | 1.2 |
| v0_s4 | 400 | 99.8 | 51.0 | 16.8 | 14.0 | 16.6 | 399 | 343 | 0 | 28.7 | 15.0 | 10.0 | 12.5 | 3.8 |
| v0_s5 | 400 | 99.8 | 60.0 | 20.2 | 17.0 | 21.2 | 399 | 331 | 0 | 27.5 | 18.8 | 23.8 | 15.0 | 0.0 |
| v2_s0 | 400 | 99.8 | 62.3 | 31.2 | 28.0 | 34.4 | 399 | 287 | 0 | 43.8 | 35.0 | 32.5 | 26.2 | 2.5 |
| v2_s1 | 400 | 100.0 | 64.0 | 33.5 | 30.2 | 37.8 | 400 | 279 | 0 | 43.8 | 26.2 | 47.5 | 33.8 | 0.0 |
| v2_s2 | 400 | 100.0 | 59.8 | 29.8 | 25.0 | 30.9 | 400 | 300 | 0 | 41.2 | 20.0 | 33.8 | 28.7 | 1.2 |
| v2_s3 | 400 | 100.0 | 61.0 | 31.2 | 29.2 | 35.9 | 400 | 283 | 0 | 51.2 | 15.0 | 46.2 | 31.2 | 2.5 |
| v2_s4 | 400 | 100.0 | 56.8 | 30.0 | 25.8 | 30.9 | 400 | 297 | 0 | 57.5 | 20.0 | 26.2 | 20.0 | 5.0 |
| v2_s5 | 400 | 100.0 | 56.8 | 26.2 | 24.2 | 30.3 | 400 | 303 | 0 | 36.2 | 5.0 | 46.2 | 33.8 | 0.0 |

## Primary: V2-arm minus V0-arm, live-verified recovery (V2), faulted kinds, 6 paired seeds
- difference: **+10.36 points**; two-level bootstrap (seeds, episodes within kind) 95% CI [+5.83, +15.26]
- paired t over seeds (conditional on these episodes): mean +10.36, sd 3.98, t=6.38 (df 5), p=0.001, 95% CI [+6.19, +14.54]
- per-seed differences: +7.5, +10.3, +5.3, +15.6, +14.4, +9.1
- MMD 7.2 points; **outcome class: positive effect, size unresolved (CI excludes 0, upper bound >= MMD)**

## Other contrasts (descriptive)
- primary_all5_pooled: +8.46 [+4.75, +12.38]
- label_accuracy_diff_v2arm_minus_v0arm_fault4: -0.10 [-5.42, +5.36]
- evidence_grounded_diff_v2arm_minus_v0arm_fault4: +10.68 [+5.94, +15.94]
- v2_arm_minus_csft (V2, fault kinds): +3.70 [-1.67, +9.11]
- v0_arm_minus_csft (V2, fault kinds): -6.67 [-12.40, -1.20]
- sft_minus_base (V2, fault kinds): +12.19 [+6.88, +17.81]
- csft_minus_sft (V2, fault kinds): +12.50 [+6.88, +18.44]
- V2-arm minus V0-arm by family (points, no CI): {"A": 8.541666666666666, "B_faulted": 12.1875, "B_OK": 0.8333333333333334}
- replay crash/timeout rollouts (counted as unverified-failure): 0

## Reading (no numbers beyond the script output above)
- Pre-registered outcome class: **positive effect, size unresolved**. The two-level bootstrap CI for V2-arm minus V0-arm (live-verified recovery, faulted kinds) excludes zero; its upper bound is above the 7.2-point minimum meaningful difference and its lower bound is below it, so the data neither show an effect of at least 7.2 points nor rule one out. Per-seed differences were positive in all 6 seeds.
- The gap is in evidence grounding, not diagnosis and not repair choice (the independent recount below shows both arms choose the recovering repair equally often): label accuracy (V0) of the two arms is the same (difference about 0), while the evidence-grounded rate (V1) and V2 differ by about the same amount. Every recovering chosen repair was the `correct`-role candidate (the menu has exactly one recovering repair per episode), so V2 is V1 plus "chose that repair"; V1 and V2 are nearly the same measure here, the design lesson already recorded in DECISIONS.
- Against continued SFT: the V2 arm is not distinguishable from csft (CI spans 0); the V0 arm is below csft (CI excludes 0). csft is a single run, so these are optimistic. Continued SFT alone recovers a large share of the GRPO gain, as in DEV.
- Claimed vs verified: nearly all valid answers claim a fix; most claims are unverified for every policy (column in the table).
- B_OK (healthy control, old prompt that presupposes a fault) stays near 0 for all GRPO arms, as on DEV.

## Caveats and honest limits
- Exploratory: two task families (A, B), one task design, 6 paired seeds (robustness minimum, not power). No claim about unseen mechanisms. The training reward was the DEV `dev_container` table; TEST scores are live sandbox only.
- csft and sft are single runs; no seed variance for them.
- Nothing was tuned on TEST. No scored result was re-run.

## Infrastructure events (all before any scoring; nothing re-sampled)
1. First build attempt: the 3-episode determinism check FAILED because each container drew a random TD_KEY and candidate/evidence ids and data salts are keyed by it. Isolated and fixed by using one fixed key for the whole TEST run; check then passed (3 episodes built twice in separate containers, identical hashes). The failed attempt produced no TEST output used later.
2. `assemble_test` first failed with ModuleNotFoundError (sys.path); re-run, no sampling involved.
3. Ledger bookkeeping: the failed first build left a projected row; replaced by an estimated actual row of 4800 cpu-s (about $0.19), an estimate not a measurement.

## Cost (ledger `actual` rows with run_id starting `test_`)
CPU build $0.50 (includes the $0.19 estimate), 15 GPU evals $2.66, replays $2.74; total about $5.90 of the $12 task cap (ledger total moved from $9.58 to about $15.5). Observed replay cost about $0.0075 per pair (ledger model, not provider billing; provider billing not checked).

## Commits
See `git log` on branch `modal-infra`: protocol e2e2560, builder/eval-mode commits, results commit below. Nothing pushed.

## Independent recount and falsification check (2026-10-19)
`scripts/audit_test_recount.py` (does not import `analyze_test.py`) recounts from the raw eval and replay JSONs; output in `docs/results/test/AUDIT_RECOUNT.txt`.
- V0 and V2 per policy, arm means, per-seed differences, paired t and the percentile bootstrap all match this report (primary +10.36, CI [+5.83, +15.26], t=6.38, p=0.0014; all-5-kinds +8.46; label -0.10; V1 +10.68). V1 is read from `test_scored.json` because the ground truth for the evidence predicate is only in the Modal volume pickle; consistency checks pass (V2 implies V1, V2 implies an ok live replay, 0 rollouts counted recovered without a replay).
- Adapter mapping: `v0_sN` evaluated `/results/g_v0[_sN]/adapter` (training JSON `arm=V0`), `v2_sN` evaluated `/results/g_v2[_sN]/adapter` (`arm=V2`), seeds 0-5 paired, same SFT init (sft1); `sft`=sft1, `csft`=csft. No swap.
- Replays: 369 unique pairs, no conflicting duplicates, all `modal_sandbox`; every ok replay is the `correct` role, one per episode. Two `fault_not_injected` pairs (episodes 22 and 43, `exclude_rank`, ok=False) were chosen by 6 rollouts (base 4, csft 1, v0_s4 1), negligible.
- Integrity: seed file sha256 equals the protocol hash and its mtime (23:09:07) precedes the protocol commit e2e2560 (23:10:52); first TEST eval finished 23:26; the TEST seeds (private; first 20 ascending per kind within the sealed pool) are disjoint from DEV (10000-10399), TRAIN (20000-20399), BURNED and from every seed in the three local episode caches; they are the first 20 ascending per kind.
- Limits: not independently recounted are V1 (needs ground truth) and the sandbox replays themselves; the generator is shared between TRAIN, DEV and TEST (split by seed, not mechanism).
- New finding: both arms choose the recovering repair equally often (71.4% vs 71.5% of faulted rollouts), so the arm difference is entirely evidence citation. Rollouts that cite wrong evidence but pick the recovering repair are 868 (V0 arm) and 676 (V2 arm) of 1920; this, not failed recovery, is most of the "claimed, not verified" count.
- DEV vs TEST: on the 4 faulted kinds DEV (unweighted kind mean) is +5.64 and TEST +10.36; all five per-kind V2 differences have the same sign on both; per-seed differences correlate 0.56 (n=6). The size difference is not isolated; possible contributors are different episodes (48 vs 100), live vs table scoring, G=8 vs 4.
- Not undermined: sign, seeds, pairing, scoring. Not shown: that recovery verification matters, or any effect beyond these two task families.
