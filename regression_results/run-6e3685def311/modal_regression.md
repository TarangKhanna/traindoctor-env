| kind | seed | case | expected | got | first rejecting invariant | exec_mode | ok |
|---|---|---|---|---|---|---|---|
| A1_src | 90005 | gold | accepted | accepted | - | modal_sandbox | yes |
| A1_src | 90005 | null | rejected | rejected | I1 no_completed_relaunch | modal_sandbox | yes |
| A1_src | 90005 | F11 | rejected | rejected | I6 world_size | modal_sandbox | yes |
| A1_src | 90005 | F03 | rejected | rejected | I1 no_completed_relaunch | modal_sandbox | yes |
| A1_cfg | 90014 | gold | accepted | accepted | - | modal_sandbox | yes |
| A1_cfg | 90014 | null | rejected | rejected | I1 no_completed_relaunch | modal_sandbox | yes |
| A1_cfg | 90014 | F11 | rejected | rejected | I6 world_size | modal_sandbox | yes |
| A1_cfg | 90014 | F03 | rejected | rejected | I1 no_completed_relaunch | modal_sandbox | yes |
| A2 | 90003 | gold | accepted | accepted | - | modal_sandbox | yes |
| A2 | 90003 | null | rejected | rejected | I1 no_completed_relaunch | modal_sandbox | yes |
| A2 | 90003 | F11 | rejected | rejected | I6 world_size | modal_sandbox | yes |
| A2 | 90003 | F03 | rejected | rejected | I1 no_completed_relaunch | modal_sandbox | yes |
| B_SKIP | 90001 | gold | accepted | accepted | - | modal_sandbox | yes |
| B_SKIP | 90001 | null | rejected | rejected | samples gaps=24 dups=24 | modal_sandbox | yes |
| B_SKIP | 90001 | F16 | rejected | rejected | samples gaps=24 dups=24 | modal_sandbox | yes |
| B_SKIP | 90001 | F09 | rejected | rejected | I4 ranks_inconsistent | modal_sandbox | yes |
| B_DUP | 90000 | gold | accepted | accepted | - | modal_sandbox | yes |
| B_DUP | 90000 | null | rejected | rejected | samples gaps=21 dups=21 | modal_sandbox | yes |
| B_DUP | 90000 | F16 | rejected | rejected | samples gaps=21 dups=21 | modal_sandbox | yes |
| B_DUP | 90000 | F09 | rejected | rejected | I4 ranks_inconsistent | modal_sandbox | yes |
| B_SKIP | 90001 | bad_config (world_size: 100000) | refused, detail 'BAD_CONFIG: world_size must be an integer in [1, 16]' | 'BAD_CONFIG: world_size must be an integer in [1, 16]' | - | modal_sandbox | yes |

commit 6e3685def311ec3f13a406994a18615751713dc4  date 2026-10-21T18:54:49+00:00  python 3.12.9  modal 1.6.2  spend (script counter) $0.100
violations: 0
