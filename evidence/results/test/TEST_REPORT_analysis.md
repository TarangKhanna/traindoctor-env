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
