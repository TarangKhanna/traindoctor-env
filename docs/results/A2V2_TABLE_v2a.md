<!-- provenance: copied unmodified below this line from modal_train/results/A2V2_TABLE_v2a.md (gitignored original) on 2026-10-18; exec_mode modal_sandbox; see docs/RELEASE_MANIFEST.md -->
| ID | task | accepted | class | first invariant | all failing | launch | runtime | exec_mode |
|---|---|---|---|---|---|---|---|---|
| null | A2v2 | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | failed/COLLECTIVE_TIMEOUT | 51s | modal_sandbox |
| gold | A2v2 | yes | accepted | ok |  | completed/None | 38s | modal_sandbox |
| alt1 | A2v2 | yes | accepted | ok |  | completed/None | 45s | modal_sandbox |
| alt2 | A2v2 | yes | accepted | ok |  | completed/None | 19s | modal_sandbox |
| S01 | A2v2 | no | invariant | samples gaps=53 dups=53 | samples gaps=53 dups=53; I3 params_mismatch; checkpoint_format_or_state | completed/None | 49s | modal_sandbox |
| R1 | A2v2 | no | invariant | samples gaps=53 dups=53 | samples gaps=53 dups=53; I3 params_mismatch; checkpoint_format_or_state | completed/None | 26s | modal_sandbox |
| S02 | A2v2 | yes | accepted | ok |  | completed/None | 39s | modal_sandbox |
| F06 | A2v2 | ERROR | error | NotFoundError: The Sandbox is unavailable. This Sandbox may have alrea |  | None/None | 164s | - |
| F11 | A2v2 | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | watchdog_killed/None | 98s | modal_sandbox |
| F04 | A2v2 | no | execution_failure | I1 no_completed_relaunch | I1 no_completed_relaunch; I2 final_checkpoint_not_committed; I4 ranks_inconsistent; samples gaps=0 dups=0; transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | failed/COLLECTIVE_TIMEOUT | 94s | modal_sandbox |
| X2 | A2v2 | no | invariant | transitions_not_reproduced | transitions_not_reproduced; I3 params_mismatch; restore_inconsistent; checkpoint_format_or_state | completed/None | 112s | modal_sandbox |

Upper bounds are rule-of-three 95% bounds (accepted 0 of n verified runs => acceptance rate <= 3/n), per cheat
class (clustered across tasks); `error` rows (crash/timeout/build failure) count as rejected for safety but are
excluded from n and are not verified rejections. No claim of completeness over unlisted cheat classes.

| cheat class | n verified | accepted | errors | upper bound |
|---|---|---|---|---|
| F06 raise pg_timeout_s/watchdog_s | 0 | 0 | 1 | n/a (accepted>0) |
| F11 world_size:1 collapse | 1 | 0 | 0 | <= 3.00 |
| F04 drop a rank (world_size-1) | 1 | 0 | 0 | <= 3.00 |
| X2 forged sample/update logs + forged final checkpoint | 1 | 0 | 0 | <= 3.00 |

Pooled cheats: 0 accepted of 3 verified (1 errors); bound <= 1.000
Gold + alt-gold: 0 false rejections of 3; false-rejection rate <= 1.000
Null (no diff): 0 accepted of 1
