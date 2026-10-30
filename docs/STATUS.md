# Local development status

2026-10-30 bounded follow-up: companion research branch `demo/masked-mean-v1`
implements one masked-loss source-repair incident using this environment's
existing launcher, per-step checkpoint copies, Tier B tools and isolation.
Environment source and historical v1.2 are unchanged. Correct controls and the
scripted tool smoke pass; neither actual Qwen/Gemma episode applies or relaunches
a repair. A separately reviewed Qwen text fragment executes and fails Python;
it is not autonomous model credit. Final evidence and reproduction commands are
in the research branch's masked-mean report. Stop without training or more models.

2026-10-29: `demo/state-recovery-v1` adds one state-only recovery demonstration in isolation from historical Task B. The original training source and historical verifier remain unchanged. The selected continuation is graded, with correctness separated from repeated optimizer work. Companion execution and independent continuation-oracle evidence are in the matching research worktree.

Full committed restoration remains a common robust action. The work targets a reproducible engineering example; it does not establish a challenging RL task. See [the protocol contract](STATE_RECOVERY_V1.md).

Final network verification isolated a pre-existing wildcard TCPStore listener. The shared launcher now owns a pre-bound loopback TCPStore; live lsof regression passes, as do 241 focused local tests and the repeated twelve-continuation demonstration. Historical v1.2 and training source remain unchanged.

Local follow-up: scoped [agent instructions](../AGENTS.md) now bind this branch to the worked example, reuse-first implementation and measured evidence. [The append-only learning log](LEARNINGS.md) records catches and limits; this documentation update adds no experiment or runtime.

2026-10-29 release preparation: README now separates historical Modal evaluation from the local recovery example, links the saved-evidence quickstart, and states that successful local multi-turn code repair is not established. Paired-source installation and saved recount were checked; core behavior and verifier v1.2 remain unchanged.

2026-10-29 focused README compression: concise recovery-first landing page; prior full content retained in docs/PROJECT_DETAIL.md. Relative links, code fences, numerical provenance and diff checks pass; read-only critic found no blocker. No new experimental claim.

2026-10-30 isolated demo/harbor-v1: one B_DUP Harbor schema1.4 task, source-only separate verifier, task-specific Docker uid/capability bridge and 18 adapter tests implemented. Native Compose/static checks pass; AI security review found no remaining static blocker. Harbor CLI absent and Docker stopped at preflight; owner approval for isolated installation/backend startup remains pending. No Harbor oracle/nop/invalid runtime verdict or overhead is claimed. Core v1.2 and historical evidence unchanged.

2026-10-30 runtime validation supersedes the pending setup above: isolated Harbor 0.25.0 / schema 1.4 and local Docker Desktop execute oracle1, no-op0, completing-invalid0 and tamper0. Final hardened controls share the same task checksum, pass 20 adapter tests and agree with native v1.2 source/verdict controls. Default lint retains one reviewed capability exception; runtime network confinement is Compose `none`, not Harbor provider policy. Exact receipts, image identities and costs are in the companion research packet `results/harbor_validation_20261030/`. Own trial containers/volumes cleaned; images/Desktop retained. GO local interoperability smoke only; no benchmark, model/RL, release or push claim. See docs/HARBOR.md.

2026-10-30 owner-authorized public release: concise README now links the validated Harbor task and measured research evidence, while keeping the unfinished autonomous-agent result explicit. Preserve v1.2 and publish only the reviewed/tested bounded work through a normal fast-forward push.
