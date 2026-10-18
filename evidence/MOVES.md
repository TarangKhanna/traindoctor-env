# Moves

Repository reorganization (branch `repo-cleanup`, base `2f1ed9c`). Every move used `git mv`, so history is preserved, and file content is byte-identical (git similarity R100) except where noted below.

## Rules applied

- `docs/` keeps only maintained, reader-facing guides and the live figures.
- `experiments/` holds agent evaluations and verifier-audit experiments (code and reports).
- `evidence/` holds raw results, frozen protocols, checksums and provenance manifests. `evidence/archive/` holds superseded specs, notes and loose development JSON.
- Hash-pinned results and frozen protocols are not edited. They may still mention pre-reorganization paths (for example `docs/results/`, `docs/prereg/`, `cd docs && shasum -c results/SHA256SUMS` in `results/SANITIZATION.md`); map them with the table below. Archived notes and `evidence/archive/writeups/MLX_MODAL_RL.md` cite development-only files (`docs/NOTES.md`, `docs/LEARNINGS.md`, `docs/PUBLISH.md`, `.research/HANDOFF.md`) that were never part of this repository; they are citations of sources, not tracked files.
- `evidence/results/SHA256SUMS` now uses paths relative to `evidence/results/` (REDTEAM_TABLE.md, A2V2_SCENARIO.md and VERIFIER_V11.md moved into that folder so one checksum file still covers them). Verify with `cd evidence/results && shasum -a 256 -c SHA256SUMS`. The sorted hash column is identical before and after.

## Content edits that are not pure moves

- README.md, docs/*.md, evidence/RELEASE_MANIFEST.md: links and path references updated; README heading and quickstart link line (commit 1).
- Historical banners marking the 2,048-token cap as historical (current cap 3,072, `CAP_TOKENS = 3072` in `traindoctor_env/bundle.py`, verified): `label_order_confound.md`, `splits.md`, `tokenizer_cap_and_menu_fix.md`, `config_visibility_note.md`, `leak_validation_*.md`. No code constant was changed; no doc/code mismatch found.
- Hardcoded path constants and comments only, no logic: `audit/menu_leak.py`, `rl/token_audit.py`, `modal_train/cpu_app.py` (comment), `scripts/{analyze_test,audit_test_recount,check_readme,make_figures,mutation_check_leak,redteam_a2v2,redteam_suite,repair_choice_by_seed}.py`, `tests/{test_analyze_test,test_config_swap,test_dataserver,test_verifier_v11,test_workspace}.py` (docstring paths), `traindoctor_env/reexec.py` (comment), `evidence/prereg/power_analysis_sim.py` (output path), `experiments/verifier_audit/*.py`.
- `scripts/make_figures.py` no longer generates `fig_two_panel`, `fig_fake_repairs`, `fig_test_results` (unreferenced by any document); `fig_failure.svg` and `fig_redteam_funnel.svg` regenerate byte-identically. `fig_reward_outcome.svg` and `fig_trust_boundary.svg` are hand-authored and were never generated.

## Personal or session files

Searched tracked files for interview, resume, LEARNINGS, job-search and owner-private paths (`/Users/...`): none tracked. The only hits are `.gitignore` patterns and source citations of non-existent private files inside the archived write-up noted above.

## Decision table

| old path | new path | reason |
|---|---|---|
| `docs/A2V2_SCENARIO.md` | `evidence/results/A2V2_SCENARIO.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/GITHUB_TOPICS.md` | `evidence/archive/notes/GITHUB_TOPICS.md` | development note or outdated metadata, superseded by maintained docs |
| `docs/REDTEAM_TABLE.md` | `evidence/results/REDTEAM_TABLE.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/RELEASE_MANIFEST.md` | `evidence/RELEASE_MANIFEST.md` | provenance manifest (claim -> file -> command); publication statement corrected |
| `docs/VERIFIER_V11.md` | `evidence/results/VERIFIER_V11.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/audits/bundle_evidence_oracle_796ba23.json` | `experiments/verifier_audit/bundle_evidence_oracle_796ba23.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/bundle_evidence_oracle_796ba23.py` | `experiments/verifier_audit/bundle_evidence_oracle_796ba23.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/config_visibility_note.md` | `experiments/verifier_audit/config_visibility_note.md` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_cfgonly.json` | `experiments/verifier_audit/indep2_cfgonly.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_cfgonly.py` | `experiments/verifier_audit/indep2_cfgonly.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_dissect.json` | `experiments/verifier_audit/indep2_dissect.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_dissect.py` | `experiments/verifier_audit/indep2_dissect.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_feats.py` | `experiments/verifier_audit/indep2_feats.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_gen.py` | `experiments/verifier_audit/indep2_gen.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_ids.json` | `experiments/verifier_audit/indep2_ids.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_ids.py` | `experiments/verifier_audit/indep2_ids.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_menu_leak_results.json` | `experiments/verifier_audit/indep2_menu_leak_results.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_models.py` | `experiments/verifier_audit/indep2_models.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_null_M.json` | `experiments/verifier_audit/indep2_null_M.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_null_M.py` | `experiments/verifier_audit/indep2_null_M.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/indep2_run.py` | `experiments/verifier_audit/indep2_run.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_796ba23.json` | `experiments/verifier_audit/leak_validation_796ba23.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_796ba23.md` | `experiments/verifier_audit/leak_validation_796ba23.md` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_796ba23.py` | `experiments/verifier_audit/leak_validation_796ba23.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_796ba23_ablation.json` | `experiments/verifier_audit/leak_validation_796ba23_ablation.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_796ba23_ablation.py` | `experiments/verifier_audit/leak_validation_796ba23_ablation.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_97a49d8.json` | `experiments/verifier_audit/leak_validation_97a49d8.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_97a49d8.md` | `experiments/verifier_audit/leak_validation_97a49d8.md` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_97a49d8.py` | `experiments/verifier_audit/leak_validation_97a49d8.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/leak_validation_97a49d8_ablation.py` | `experiments/verifier_audit/leak_validation_97a49d8_ablation.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/menu_leak_before_after.json` | `experiments/verifier_audit/menu_leak_before_after.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/menu_leak_before_after.py` | `experiments/verifier_audit/menu_leak_before_after.py` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/menu_leak_dev400.json` | `experiments/verifier_audit/menu_leak_dev400.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/mutation_check_leak.json` | `experiments/verifier_audit/mutation_check_leak.json` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/audits/token_cap_3072.md` | `experiments/verifier_audit/token_cap_3072.md` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |
| `docs/baseline_pilot16_summary.json` | `evidence/archive/dev_json/baseline_pilot16_summary.json` | loose development output, not referenced by any maintained doc |
| `docs/benchmarks/cross_agent_dev6/*` (72 files) | `experiments/cross_agent_dev6/*` | agent evaluation experiment; raw answers, replays and SHA256SUMS moved together |
| `docs/control_evidence_only.json` | `evidence/archive/dev_json/control_evidence_only.json` | loose development output, not referenced by any maintained doc |
| `docs/control_menu_only.json` | `evidence/archive/dev_json/control_menu_only.json` | loose development output, not referenced by any maintained doc |
| `docs/control_neutral.json` | `evidence/archive/dev_json/control_neutral.json` | loose development output, not referenced by any maintained doc |
| `docs/figures/fig_fake_repairs.svg` | `(deleted)` | generated figure referenced by no document; removed from scripts/make_figures.py too |
| `docs/figures/fig_test_results.svg` | `(deleted)` | generated figure referenced by no document; removed from scripts/make_figures.py too |
| `docs/figures/fig_two_panel.svg` | `(deleted)` | generated figure referenced by no document; removed from scripts/make_figures.py too |
| `docs/label_order_confound.md` | `evidence/archive/notes/label_order_confound.md` | development note or outdated metadata, superseded by maintained docs |
| `docs/label_shuffle_q4b_i.json` | `evidence/archive/dev_json/label_shuffle_q4b_i.json` | loose development output, not referenced by any maintained doc |
| `docs/m0/fault_proof.md` | `evidence/archive/m0/fault_proof.md` | M0 fault proof note, historical |
| `docs/model_screen.md` | `evidence/archive/notes/model_screen.md` | development note or outdated metadata, superseded by maintained docs |
| `docs/prereg/amendment_token_cap.md` | `evidence/prereg/amendment_token_cap.md` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/prereg/power_analysis_amendment.md` | `evidence/prereg/power_analysis_amendment.md` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/prereg/power_analysis_results.json` | `evidence/prereg/power_analysis_results.json` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/prereg/power_analysis_sim.py` | `evidence/prereg/power_analysis_sim.py` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/prereg/strata_config_baseline.json` | `evidence/prereg/strata_config_baseline.json` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/prereg/strata_config_baseline.py` | `evidence/prereg/strata_config_baseline.py` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/prereg/strata_v0_v2.md` | `evidence/prereg/strata_v0_v2.md` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/prereg/test_protocol.md` | `evidence/prereg/test_protocol.md` | frozen protocol or pre-registration (immutable); only power_analysis_sim.py output path updated |
| `docs/probe_q4b_i.json` | `evidence/archive/dev_json/probe_q4b_i.json` | loose development output, not referenced by any maintained doc |
| `docs/quickstart_local_output.txt` | `evidence/quickstart_local_output.txt` | recorded output, evidence rather than a guide |
| `docs/results/A2V2_TABLE_v2a.md` | `evidence/results/A2V2_TABLE_v2a.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/A2V2_TABLE_v2b_f06.md` | `evidence/results/A2V2_TABLE_v2b_f06.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/CONTROLS_REPORT.md` | `evidence/results/CONTROLS_REPORT.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/CONTROLS_SUMMARY.json` | `evidence/results/CONTROLS_SUMMARY.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/LEDGER_ACTUAL_SUMMARY.txt` | `evidence/results/LEDGER_ACTUAL_SUMMARY.txt` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/MODAL_SANDBOX_PROOF.md` | `evidence/results/MODAL_SANDBOX_PROOF.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/PILOT_REPORT.md` | `evidence/results/PILOT_REPORT.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/REDTEAM_TABLE_v11.md` | `evidence/results/REDTEAM_TABLE_v11.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/SANITIZATION.md` | `evidence/results/SANITIZATION.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/SEEDS_3_5_REPORT.md` | `evidence/results/SEEDS_3_5_REPORT.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/SHA256SUMS` | `evidence/results/SHA256SUMS` | checksum file: only the path column changed; hash column identical |
| `docs/results/TEST_REPORT.md` | `evidence/results/TEST_REPORT.md` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/exploratory_v2/*` (16 files) | `evidence/results/exploratory_v2/*` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/modal_sandbox_proof.json` | `evidence/results/modal_sandbox_proof.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/quickstart_verified_replay.txt` | `evidence/results/quickstart_verified_replay.txt` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/redteam_a2v2_v2a.json` | `evidence/results/redteam_a2v2_v2a.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/redteam_a2v2_v2b_f06.json` | `evidence/results/redteam_a2v2_v2b_f06.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/redteam_isolate.json` | `evidence/results/redteam_isolate.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/redteam_isolate_long.json` | `evidence/results/redteam_isolate_long.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/redteam_modal_ERRORED_403_run.json` | `evidence/results/redteam_modal_ERRORED_403_run.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/redteam_modal_v11.json` | `evidence/results/redteam_modal_v11.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/test/*` (26 files) | `evidence/results/test/*` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/v1v2_seeds_out.json` | `evidence/results/v1v2_seeds_out.json` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/results/v1v2_seeds_stdout.txt` | `evidence/results/v1v2_seeds_stdout.txt` | raw result or report, hash-pinned by SHA256SUMS (bytes unchanged) |
| `docs/screen_q4b.json` | `evidence/archive/dev_json/screen_q4b.json` | loose development output, not referenced by any maintained doc |
| `docs/screen_q4b_i.json` | `evidence/archive/dev_json/screen_q4b_i.json` | loose development output, not referenced by any maintained doc |
| `docs/specs/M1.md` | `evidence/archive/specs/M1.md` | old milestone spec, superseded |
| `docs/specs/M1_5.md` | `evidence/archive/specs/M1_5.md` | old milestone spec, superseded |
| `docs/specs/M1_FINISH.md` | `evidence/archive/specs/M1_FINISH.md` | old milestone spec, superseded |
| `docs/specs/M1_NOTES.md` | `evidence/archive/specs/M1_NOTES.md` | old milestone spec, superseded |
| `docs/specs/M1_v2.md` | `evidence/archive/specs/M1_v2.md` | old milestone spec, superseded |
| `docs/specs/M2_verifiers.md` | `evidence/archive/specs/M2_verifiers.md` | old milestone spec, superseded |
| `docs/splits.md` | `evidence/archive/notes/splits.md` | development note or outdated metadata, superseded by maintained docs |
| `docs/token_audit.json` | `evidence/archive/dev_json/token_audit.json` | loose development output, not referenced by any maintained doc |
| `docs/tokenizer_cap_and_menu_fix.md` | `evidence/archive/notes/tokenizer_cap_and_menu_fix.md` | development note or outdated metadata, superseded by maintained docs |
| `docs/writeups/MLX_MODAL_RL.md` | `evidence/archive/writeups/MLX_MODAL_RL.md` | redundant infrastructure/RL write-up citing private files; kept for the record |
| `docs/writeups/README.md` | `evidence/archive/writeups/README.md` | redundant infrastructure/RL write-up citing private files; kept for the record |
| `docs/writeups/VERIFIER_AUDIT.md` | `experiments/verifier_audit/VERIFIER_AUDIT.md` | verifier-audit experiment code and reports (menu-leak, bundle evidence, token cap) |

## Retained at old path

None. Every evidence, protocol, checksum and manifest reference was updated and verified (see the checks recorded in the commit messages of this branch); no file was left behind in `docs/` other than the maintained guides.
