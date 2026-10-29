# Local development status

2026-10-29: `demo/state-recovery-v1` adds one state-only recovery demonstration in isolation from historical Task B. The original training source and historical verifier remain unchanged. The selected continuation is graded, with correctness separated from repeated optimizer work. Companion execution and independent continuation-oracle evidence are in the matching research worktree.

Full committed restoration remains a common robust action. The work targets a reproducible engineering example; it does not establish a challenging RL task. See [the protocol contract](STATE_RECOVERY_V1.md).

Final network verification isolated a pre-existing wildcard TCPStore listener. The shared launcher now owns a pre-bound loopback TCPStore; live lsof regression passes, as do 241 focused local tests and the repeated twelve-continuation demonstration. Historical v1.2 and training source remain unchanged.

Local follow-up: scoped [agent instructions](../AGENTS.md) now bind this branch to the worked example, reuse-first implementation and measured evidence. [The append-only learning log](LEARNINGS.md) records catches and limits; this documentation update adds no experiment or runtime.

2026-10-29 release preparation: README now separates historical Modal evaluation from the local recovery example, links the saved-evidence quickstart, and states that successful local multi-turn code repair is not established. Paired-source installation and saved recount were checked; core behavior and verifier v1.2 remain unchanged.

2026-10-29 focused README compression: concise recovery-first landing page; prior full content retained in docs/PROJECT_DETAIL.md. Relative links, code fences, numerical provenance and diff checks pass; read-only critic found no blocker. No new experimental claim.
