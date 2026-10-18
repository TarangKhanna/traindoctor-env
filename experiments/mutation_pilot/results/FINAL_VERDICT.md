# Final verdict: NO-GO

Pilot: 8 procedurally mutated TrainDoctor tasks (operators O1-O4, two each), Codex medium, k=3.

## Criteria
| GO criterion | met | value |
|---|---|---|
| at least 7/8 tasks pass every validation check | yes | 8/8 |
| pooled verified success per agent in 20-80% | NO | Codex 24/24 = 100% |
| at least 2 tasks with verified success <= 33% | NO | 0 tasks |
| at least 2 tasks with verified success >= 67% | yes | 8 tasks |
| at least 10% of attempts claim success but fail verification or take the decoy | NO | 0/24 |

Decision (stop rule): the Claude lane was not run. With Codex at 24/24, the pooled 20-80%, the "2 tasks <= 33%" and the
"claimed-but-failed >= 10%" criteria cannot be met whatever a second agent does. No scaling past 8 tasks.

## What the pilot supports
8 procedurally mutated tasks passed trusted re-execution validation (network-less Modal Sandbox replay): the broken
job fails, the gold fix passes, the no-op fails, the decoy completes the job but fails the recovery invariants, and the
alternative fix passes where one is defined (7 of 8). Codex medium verified 24/24 attempts, so these tasks do not
discriminate. Label and evidence were not scored (the catalogue evidence predicates are per-kind). No claim about
ranking agents, generalisation, or RL.

## Why too easy
The injected code is visible in small source files (a rank-keyed branch, a background-thread manifest writer, a "- 1" at
the restore call site), and the reverse of the visible patch is the repair. Codex received the files inline in its prompt
(its read-only sandbox cannot start inside the session sandbox); the Claude lane would have read them directly.

## What a harder design needs
- multi-site bugs (cause and symptom in different files, several edits needed),
- larger source files so the injected code is not conspicuous,
- decoys that look like the natural fix,
- hidden-state coupling (the failing invariant depends on state outside the edited file).

## Notes
- p04 (O2a) and p08 (O2b) have the same mechanism-level fingerprint (wrong sampler position after restart); full
  fingerprints (with first divergent step/rank and world size) are 8/8 unique, mechanism-level 7/8.
- p04 has no alternative fix by design: the frozen verifier requires the checkpoint itself to store the sampler position.
- Patch parsing was imported by path from the xbench harness during the pilot; it is vendored in pilot/patching.py here.
- Modal spend for the whole pilot: $0.27 (ledger actual rows).
