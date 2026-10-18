# Evidence

Raw results, frozen protocols, and the commands that recompute each headline number. Run commands from the repository root with the repo environment (`pip install -r requirements-cpu.txt`) and `PYTHONPATH=. TD_KEY=<any hex>`.

| path | contents |
| --- | --- |
| [`results/`](results/) | sealed-TEST scored rollouts and live replay records (`test/`), red-team tables, reports; `cd evidence/results && shasum -a 256 -c SHA256SUMS` checks every hash-pinned file |
| [`prereg/`](prereg/) | preregistration, power-analysis amendment, token-cap amendment and the sealed-TEST protocol, written before the runs they govern |
| [`quickstart_local_output.txt`](quickstart_local_output.txt) | recorded output of the local quickstart |

## Headline numbers and how to recompute them

| claim (README) | file | recompute |
| --- | --- | --- |
| Held-out full-task success: base 5.0%, SFT 17.2%, continued SFT 29.7%, GRPO arms 23.0% / 33.4%; stronger-reward minus diagnosis-only +10.36 points, 95% bootstrap interval +5.83 to +15.26; versus continued SFT +3.7 (-1.7 to +9.1) | [`results/TEST_REPORT.md`](results/TEST_REPORT.md), [`results/test/TEST_REPORT_analysis.json`](results/test/TEST_REPORT_analysis.json) | `python scripts/analyze_test.py evidence/results/test/test_scored.json /tmp/an` writes `/tmp/an.json` and `/tmp/an.md`; the JSON is identical to the committed analysis |
| Working repair chosen at 71.4% vs 71.5% | [`results/test/AUDIT_RECOUNT.txt`](results/test/AUDIT_RECOUNT.txt) | `python scripts/audit_test_recount.py` and `python scripts/repair_choice_by_seed.py` |
| 369 unique scenario/repair pairs replayed in Modal Sandboxes | `results/test/test_replay_r*.json` | counted by `scripts/audit_test_recount.py` |
| Fake repairs: 0 of 86 accepted (48 invariant, 38 execution failure); reference repairs 5/5; alternative repairs 9/10; no-op 0/5; F11 leaves 72 missing and 8 duplicated samples | [`results/REDTEAM_TABLE.md`](results/REDTEAM_TABLE.md) | `awk` recount below; `python scripts/make_figures.py` regenerates the funnel figure from the same table |
| Verifier v1.1 removes the one false rejection (alternative repairs 10/10, cheats 0 of 85 verified accepted) | [`results/REDTEAM_TABLE_v11.md`](results/REDTEAM_TABLE_v11.md), [`results/redteam_modal_v11.json`](results/redteam_modal_v11.json), [`results/VERIFIER_V11.md`](results/VERIFIER_V11.md) | same `awk` command on the v1.1 table |
| V1 and V2 agree on 97.7% of 3072 development rollouts | [`results/v1v2_seeds_out.json`](results/v1v2_seeds_out.json) | `python -c "import json;d=json.load(open('evidence/results/v1v2_seeds_out.json'))['part1'];print(d['n'],d['frac_equal'],d['v1_not_v2'])"` |
| Six-seed development comparison, +4.64 points, inconclusive; pilot and continued-SFT control | [`results/SEEDS_3_5_REPORT.md`](results/SEEDS_3_5_REPORT.md), [`results/PILOT_REPORT.md`](results/PILOT_REPORT.md), [`results/CONTROLS_REPORT.md`](results/CONTROLS_REPORT.md); inputs in `results/exploratory_v2/` | the six per-seed differences are recomputed in `tests/test_analyze_test.py::test_paired_t_known_sample`; the analysis scripts that produced the reports are archived privately and not published |
| Sandbox properties (network blocked, loopback Gloo works) and the A2v2 scenario red team | [`results/MODAL_SANDBOX_PROOF.md`](results/MODAL_SANDBOX_PROOF.md), [`results/A2V2_SCENARIO.md`](results/A2V2_SCENARIO.md), `results/A2V2_TABLE_*.md`, `results/redteam_a2v2_*.json` | `python scripts/modal_sandbox_proof.py A2`, `python scripts/redteam_a2v2.py` (both need a Modal account and spend credit) |
| Minimum meaningful difference 7.2 points | [`prereg/power_analysis_amendment.md`](prereg/power_analysis_amendment.md) | `python evidence/prereg/power_analysis_sim.py 300` (slow; overwrites `power_analysis_results.json`) |
| Local quickstart and gold-versus-fake replay | [`quickstart_local_output.txt`](quickstart_local_output.txt), [`results/quickstart_verified_replay.txt`](results/quickstart_verified_replay.txt) | `python scripts/quickstart.py --fake F11 --out quickstart_out` |
| Both agents recover 6/6 development tasks (verified full-task success: Claude 5/6, Codex 6/6); Codex 24/24 on the mutation pilot | [`../experiments/cross_agent_dev6/`](../experiments/cross_agent_dev6/), [`../experiments/mutation_pilot/`](../experiments/mutation_pilot/) | each folder has its own `SHA256SUMS`; prompts, raw answers and replay outcomes of the six-task run are archived privately, not published |
| README numbers match their sources | `README.md` | `python scripts/check_readme.py` |

Red-team recount from the committed table (the raw v1 JSON was overwritten by a later errored run; this table is the surviving record):

```bash
awk -F'|' 'NF==11 && $2!~/ID|---/ {id=$2; gsub(/ /,"",id); acc=$4; gsub(/ /,"",acc); cl=$5; gsub(/ /,"",cl);
  r=(id=="null")?"null":(id=="gold")?"gold":(id~/^alt/)?"alt":(id~/^S[0-9]/)?"borderline":"cheat"; n[r" "acc" "cl]++}
  END{for(k in n)print k, n[k]}' evidence/results/REDTEAM_TABLE.md | sort
```

Expected: `alt no invariant 1`, `alt yes accepted 9`, `borderline yes accepted 1`, `cheat no execution_failure 38`, `cheat no invariant 48`, `gold yes accepted 5`, `null no execution_failure 3`, `null no invariant 2`.

Every row of both red-team tables ran with `exec_mode = modal_sandbox`: the repair executed in a network-less gVisor Modal Sandbox and the invariants were recomputed outside it. Results labelled `dev_container` came from trusted pinned code in a local container and are exploratory training-reward views only. The 38 execution failures show that those cheats do not produce a working job, not that an invariant would catch them.

The isolation runs `results/redteam_isolate*.json` are the surviving raw red-team records. The raw v1 red-team JSON, the distractor-menu gate artifacts, early specs and development notes are archived privately and not published.

Files hash-pinned by `results/SHA256SUMS` and the frozen protocols are byte-identical to their original versions, so they may mention paths and tooling from the original development layout.
