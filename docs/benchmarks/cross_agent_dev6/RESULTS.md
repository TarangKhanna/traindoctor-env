# TrainDoctor cross-agent smoke benchmark (DEV only)

Status: exploratory smoke, 6 tasks, 1 attempt per agent per task. Comparison is **controlled on inputs and budgets; isolation is instruction-based for Claude vs sandbox-by-construction for Codex.** DEV seeds 10000-10047 only; no sealed TEST data, no unseen-mechanism claim.

## Setup
| | Claude | Codex |
|---|---|---|
| model | claude-sonnet-5-5, medium effort | gpt-6.1-sol, medium (bridge `--tier medium`) |
| harness | in-session subagent of the owner's Claude Code subscription | `codex_bridge.sh`, ChatGPT subscription auth |
| tools | Read of its one prompt file, Write of its answer (instruction-based limit, Read is not path-restricted) | none (task inline, empty cwd, `--sandbox read-only`) |
| input | `prompts/tNN.md` (identical text for both agents) | same file |
| attempts | 1 per task | 1 per task |
| billing | subscription, not metered; no API keys set | subscription, not metered |

Protocol: the agent sees only the job source/config and the incident records (ids, kind, rank, step, text) and returns one JSON object {label, evidence_ids, fix_claim, patch (unified diff)}. The patch is applied to a pristine copy (editable files only: job/train.py, data.py, ckpt.py, config.yaml; other paths are flagged and discarded), then verified by live re-execution with `verify_fix`-equivalent `Replay(backend='modal')` (Modal Sandbox, gVisor, network-less, `exec_mode=modal_sandbox`). V0 = label correct; V1 = V0 + cited evidence satisfies the canonical evidence predicate (cap 8 ids, all observed); V2 = V1 + sandbox-verified recovery with preserved correctness + safe action. Controls: null (right label, no evidence, no patch) and gold (oracle label/evidence/repair). Evidence scorer, verifier and fixtures are the canonical `modal-infra` code (b6c1a0d), unmodified.

Tasks (private map; agents saw neutral ids): t01 A1 src (seed 10032), t02 A1 cfg (10001), t03 A2 (10006), t04 B_SKIP (10000), t05 B_DUP (10008), t06 B_OK / healthy (10002).

## Results
| agent | n | infra fail | format ok | V0 label | V1 evidence | V2 verified | verified recovery only | claimed fixed | overclaim | flagged/invalid attempts |
|---|---|---|---|---|---|---|---|---|---|---|
| claude_sonnet | 6 | 0 | 6/6 | 6/6 [0.61,1.00] | 5/6 [0.44,0.97] | 5/6 [0.44,0.97] | 6/6 [0.61,1.00] | 6/6 | 0/6 | 0/6 |
| codex | 6 | 0 | 6/6 | 6/6 [0.61,1.00] | 6/6 [0.61,1.00] | 6/6 [0.61,1.00] | 6/6 [0.61,1.00] | 6/6 | 0/6 | 0/6 |
| gold | 6 | 0 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 | 0/6 |
| null | 6 | 0 | 6/6 | 6/6 [0.61,1.00] | 0/6 [0.00,0.39] | 0/6 [0.00,0.39] | 0/6 | 0/6 | 0/6 | 0/6 |

Wilson 95% intervals. "Claimed fixed" = non-empty patch or NO_FAULT label; overclaim = claimed fixed but not verified.

Per task (V0 V1 V2): 
| task | kind | claude_sonnet | codex | gold | null |
|---|---|---|---|---|---|
| t01 | A1 src | 111 | 111 | 111 | 100 |
| t02 | A1 cfg | 111 | 111 | 111 | 100 |
| t03 | A2 | 111 | 111 | 111 | 100 |
| t04 | B_SKIP | 111 | 111 | 111 | 100 |
| t05 | B_DUP | 100 | 111 | 111 | 100 |
| t06 | B_OK | 111 | 111 | 111 | 100 |

Evidence citations: Codex 3-7 ids per task, Claude 5-8. Claude's only V2 miss (t05, B_DUP) is an evidence failure, not a repair failure: its patch (save the sampler state in the checkpoint) recovered under live replay with samples, restores, ranks and lost work all consistent (verified recovery true), but its 8 cited ids did not contain the pair of replayed-sample records the B_DUP predicate requires (V1 false).

Convergence: in 5 of 6 tasks (t01-t04, t06) the two agents' patches normalised to byte-identical edits (verified by the outcome cache hit; t06 both empty). Only t05 differed.

Latency (agent time, not including verification): Codex 13.0-22.2 s per task, mean 16.7 s (recorded by the harness). Claude per-task durations were not recorded by the harness; the subagent notifications reported approximately t02 12.4 s, t01 16.3 s, t03 18.7 s, t04 20.5 s, t05 22.0 s, t06 24.0 s (include tool calls; approximate). Live verification: ~19-33 s per distinct edit on Modal.

Failures / exclusions: no infrastructure failures, no unparsable answers, no patch-apply failures, no out-of-scope edits, no world_size/timeout/exit-0 edits detected (static tags plus verifier `tampered`: none). Two earlier Codex attempts were discarded as harness failures, not agent failures: (1) first prompt wording told the agent it could not run commands, so it declined to read files; (2) the read-only bridge sandbox cannot start inside the nested sandbox (`sandbox-exec` not permitted), so file reading failed. Both were replaced by the inline-prompt protocol above, applied identically to both agents (raw outputs kept as `out/run1_old_prompt`, `out/run2_sandbox_fail` in the work dir, not in the public bundle).

Cost: incremental Modal spend from ledger rows tagged `xbench-` = $0.0625 (17 actual rows, cpu-hour rate x 4 cpus x wall time); cap was $2. No unbooked rows found, so none appended. Replays of identical edits are cached and not re-run. Subscription usage (Claude Code, ChatGPT) is not metered by this harness.

## What the benchmark can and cannot distinguish
- It separates a real repair from a non-repair: null controls score V0 only (label guessed from the answer is not enough), gold scores 6/6. The decomposition is informative: V0 6/6 for every agent and null, V1 separates evidence-citing from label-only (null 0/6), V2 adds live recovery.
- It does **not** separate Claude Sonnet from Codex on these tasks. Both are at or near the ceiling; the only difference (t05) is an evidence-selection miss, one task, intervals overlap heavily (e.g. V2 5/6 vs 6/6), and no significance can be claimed. We do not claim either agent is better.
- Why: the DEV tasks have single-site, localisable bugs whose minimal fix is a few lines; both agents converge to the same edit. They do not require competing interventions, restart-is-wrong judgement, or multi-file repairs.
- Not tested: unseen mechanisms, sealed TEST, tool-using (iterative) agents, multiple attempts / pass^k, exploit behaviour (no agent attempted any), Opus.
- Design change needed for a discriminating benchmark: tasks with several plausible-but-wrong minimal fixes (decoys), faults whose evidence is spread across records, repairs spanning more than one file, cases where the surface fix passes a short run but fails sample accounting, and several attempts per task.

## Conclusion (publication-ready)
On six DEV-split TrainDoctor tasks (two collective-divergence variants, a pre-collective rank stall, a skipped-sample and a duplicated-sample checkpoint/sampler fault, and a healthy control), one attempt each at medium reasoning effort, Claude Sonnet and Codex (gpt-6.1-sol) both produced valid answers and repairs that recovered under live re-execution in a network-less Modal Sandbox in 6/6 tasks. Counting the full ladder, Codex scored V2 on 6/6 and Claude Sonnet on 5/6, the single miss being a citation set that omitted the required replayed-sample records although its patch recovered correctly; null (no-repair) controls scored 0/6 and gold 6/6. Their patches were identical in five of six tasks. With six tasks the intervals overlap broadly, the tasks sit at the ceiling, and no ranking is supported. The comparison is controlled on inputs and budgets, but isolation is instruction-based for Claude and sandbox-by-construction for Codex, and the result concerns DEV scenarios only, with no claim about unseen mechanisms. The benchmark verifies repairs reliably but needs harder, decoy-laden tasks to discriminate frontier agents.

## Reproduce
```
cd <xbench dir>; PY=<repo>/.venv/bin/python   # torch 2.14.1; TD_KEY in out/td_key.txt
$PY -I xbench.py build                          # snapshots (local, ~14 s each) + workspaces + prompts/
# Codex lane (ChatGPT auth; SSL_CERT_FILE=/etc/ssl/cert.pem; hosts chatgpt.com, *.openai.com):
$PY -I xbench.py run-codex
# Claude lane: give each prompts/tNN.md to one subagent (model claude-sonnet-5-5, effort medium), save final message to answers/claude_sonnet/tNN.json
# Scoring (Modal; TD_MODAL_SITE=<modalenv>/lib/python3.12/site-packages; hosts api.modal.com, *.modal.com, *.modal.host, *.modal-storage.com):
$PY -I xbench.py controls
$PY -I xbench.py score-all codex; $PY -I xbench.py score-all claude_sonnet
$PY -I xbench.py table
```
Artifacts: answers/, out/results/, out/outcomes/, out/table.md, public/ (sanitized bundle).
