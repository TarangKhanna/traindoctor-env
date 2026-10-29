# TrainDoctor environment: current local work

## Goal and scope

Produce a small, reproducible recovery-engineering artifact with trustworthy
execution evidence. Read [the contract](docs/STATE_RECOVERY_V1.md) and
[current status](docs/STATUS.md) first. This branch contains one worked state-only
incident pair, not a new benchmark or task-generation framework.

- Keep main worktrees, historical evidence and verifier v1.2 unchanged.
- Preserve correct scratch recovery and full restoration as legitimate actions.
  Report semantic correctness, checkpoint continuity and repeated work separately.
- Grade the actual continuation when studying recovery-state decisions. Historical
  fresh-job verification answers a different question; do not silently reinterpret it.
- A state-rule baseline matching the oracle confirms observability. A common
  optimal action limits claims about conditional decision-making.
- Do not manufacture task difficulty by rejecting a correct robust repair.

## Minimal implementation

- Reuse existing launcher, manifest, checkpoint and verifier behavior where valid.
  Prefer stdlib and installed, maintained libraries over new infrastructure.
- Before adding a wrapper, runtime, dependency or framework, write the concrete
  missing capability and why an existing API cannot supply it. Dependency changes
  still require explicit authorization. Avoid generalizing a single example.
- Keep model/runtime imports outside core environment code. No model in the oracle
  or verifier, and no hidden fault labels in observations.
- Preserve genuine committed captures; derive delivered incidents separately.
  Check source hashes, manifest validity and prefix-compatible data schedules.
- Never execute arbitrary generated code based on a homegrown validator. The
  existing reviewed-proposal demonstration accepts only its exact reviewed edit.

## Local execution and evidence

- Local inference/training is authorized in this session, subject to a useful
  scientific question and resource guards; no paid/cloud execution or publication.
  No additional experiment follows automatically from this authorization.
- One model process at a time; check existing processes and keep at least 8 GiB
  available. Bind services to loopback and verify live listeners with `lsof`;
  `MASTER_ADDR=127.0.0.1` alone did not constrain TCPStore binding.
- Record backend, platform, execution mode, source revision and configuration.
  Distinguish sandboxed continuations from trusted unsandboxed reference code.
- Preflight commands from repository docs/config; baseline before behavior changes.
  Use narrow tests first, then exercise the real smoke path. Report output, failures
  and unverified boundaries. Do not rerun expensive jobs for documentation edits.

## Record learning as work proceeds

Append verified catches to [the learning log](docs/LEARNINGS.md), including evidence,
consequence and limits. Record failed approaches without inventing a root cause.
Append changed decisions to [the decisions log](docs/DECISIONS.md), update status at
milestone closure, and correct old claims by explicit superseding entries.
"Newly observed here" is not a claim of scientific novelty. Do not edit receipts
to match a preferred explanation or turn documentation upkeep into infrastructure.

## Current isolated Harbor milestone (2026-10-30)

The owner authorizes one Harbor compatibility task, not a framework or publication.
Read docs/HARBOR.md. Preserve v1.2 and keep transferred agent source separate from
trusted execution records. Explicit harbor_docker_uid provenance requires actual
worker/isolation proofs, not a container label. The owner approved isolated CLI installation and local Docker startup on
2026-10-30. Complete only local Harbor validation; the latest instruction prohibits
push/publication. Do not run Harbor from a repository directory or allow automatic
dotenv reads. No model inference/training or new tasks in this validation session.
