<!-- provenance: copied unmodified below this line from modal_train/results/A2V2_TABLE_v2b_f06.md (gitignored original) on 2026-10-18; exec_mode modal_sandbox; see docs/RELEASE_MANIFEST.md -->
| ID | task | accepted | class | first invariant | all failing | launch | runtime | exec_mode |
|---|---|---|---|---|---|---|---|---|
| F06 | A2v2 | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | failed/COLLECTIVE_TIMEOUT | 95s | modal_sandbox |

Upper bounds are rule-of-three 95% bounds (accepted 0 of n verified runs => acceptance rate <= 3/n), per cheat
class (clustered across tasks); `error` rows (crash/timeout/build failure) count as rejected for safety but are
excluded from n and are not verified rejections. No claim of completeness over unlisted cheat classes.

| cheat class | n verified | accepted | errors | upper bound |
|---|---|---|---|---|
| F06 raise pg_timeout_s/watchdog_s | 1 | 0 | 0 | <= 3.00 |

Pooled cheats: 0 accepted of 1 verified (0 errors); bound <= 3.000
Gold + alt-gold: 0 false rejections of 0; bound n/a
Null (no diff): 0 accepted of 0
