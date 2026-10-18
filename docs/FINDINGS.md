# What the experiments taught us

The main design lesson is about reward information. A recovery-verified reward adds information over a label-plus-evidence reward only when menus contain tempting repairs that complete the run but are wrong. In the v1 menus each episode has exactly one recovering repair, so V2 is V1 plus "chose that repair". Over 3072 DEV rollouts V1 and V2 agree on 97.7% (they differ on 2.3%; [`docs/writeups/VERIFIER_AUDIT.md`](writeups/VERIFIER_AUDIT.md) section 6). That matches the sealed TEST result: the verified reward helped the model cite better evidence, and did not change which repair it chose.

Next: menus with at least two label-consistent options that complete the run, each distractor violating exactly one V2 invariant, with outcomes from live re-execution ([`docs/DESIGN_NOTES.md`](DESIGN_NOTES.md)). The A2v2 scenario ([`docs/A2V2_SCENARIO.md`](A2V2_SCENARIO.md)) is a first step in that direction.

DEV split, 6 paired seeds: V2 arm minus V0 arm = +4.64 points, 95% CI [-0.69, +9.98], inconclusive.

Spend: about $15.5 of Modal credit in total at the time of writing (ledger not published).
