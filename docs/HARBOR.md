# One Harbor compatibility task

Status: **packaging implemented; Harbor execution not yet established**. This is
one existing checkpoint/sampler code-repair integration smoke test, not a new
research benchmark. Historical verifier v1.2 and original evidence are unchanged.

## What is reused

`harbor/tasks/training-repair-01/` uses the existing two-rank CPU/Gloo runtime,
checkpoint records and semantic verifier. A prepared incident supplies logs and
real source. The agent edits `/submission/ckpt.py`; only that file crosses into a
fresh, separate verifier. The oracle repairs the checkpoint writer. A no-op and a
repair that completes while preserving incorrect sample semantics must fail.
Those are acceptance criteria, not claims of executed Harbor results.

The task deliberately retains the original fresh-source re-execution objective:
check whether the proposed code makes a subsequent preemption/restart correct.
It does not claim that the agent's own diagnostic run is the final scored
continuation. The local agent-v2 research protocol studies that separate question.
This smoke task retains the known source-mutation shortcut and cannot support a
claim of a difficult investigation benchmark or RL benefit.

## Source and layout

Compatibility was inspected against official Harbor commit
`509b119db0d97b141dcd9c80f4f7ab861c47c2d6` (package 0.25.0, schema 1.4).

```text
harbor/tasks/training-repair-01/
  task.toml                 separate verifier; code-only artifact declaration
  instruction.md            actual source repair and diagnostic execution
  environment/Dockerfile    existing runtime and public incident
  environment/docker-compose.yaml
  tests/Dockerfile          fresh trusted runtime/fixture/grader
  tests/docker-compose.yaml
  tests/test.sh             preserves grader errors instead of inventing rewards
  tests/grade.py             semantic grading / explicit isolation preflight
  solution/solve.sh          oracle smoke-test intervention
```

Default Harbor Docker builds see only `environment/`. This task uses Harbor's
supported compose override to build from the repository root (`../../../..`).
Dockerfile-specific `.dockerignore` files admit only the existing runtime,
project dependency declarations and task build files. Final exclusions deny
credential paths, Git data, results, caches and virtual environments. No runtime
copy is vendored into the task. **Keep the task at its current repository path**;
it is not a standalone Harbor Hub package.

The Dockerfiles install the exact runtime dependencies from `pyproject.toml`.
Torch uses only the official CPU wheel index; other pinned runtime dependencies
use PyPI separately. An unavailable CPU wheel fails the build, with no CUDA or
version fallback. Both containers limit process counts to 256. The agent image
deletes the private fixture in the same build step after copying public evidence;
the fresh verifier alone retains its root-protected scoring fixture.
Image construction may need public package/image downloads; it does not download
model weights or use paid compute. The Python base image currently has a tag,
not a verified digest. Freeze the actual built image digest in reproduction
receipts before claiming a reproducible executed release.

## Commands after local prerequisites are available

Use a local Docker daemon, Harbor 0.25.0 and sufficient memory. No model/API is
needed for these controls. Run from a safe, non-secret working directory, passing
an absolute task path, and keep concurrency at one:

```bash
TASK=/absolute/path/to/traindoctor-env/harbor/tasks/training-repair-01
PYTHON_DOTENV_DISABLED=1 harbor lint "$TASK" --json
PYTHON_DOTENV_DISABLED=1 harbor run -p "$TASK" -a oracle -e docker -n 1 -k 1 -r 0
PYTHON_DOTENV_DISABLED=1 harbor run -p "$TASK" -a nop -e docker -n 1 -k 1 -r 0
```

`harbor lint` is the offline structural/security checker. `harbor check` invokes
an LLM quality rubric and is outside this project's no-hosted-inference scope;
old `harbor tasks check` commands are removed in this inspected version. Harbor
autoloads dotenv files in some code paths: keep `PYTHON_DOTENV_DISABLED=1`, whose
support was confirmed in the prepared python-dotenv installation, and use a clean
working directory. Never pass credential files or host API secrets to this task. The
built-in no-op name is `nop`, not `noop`.

The official verifier reads `/logs/verifier/reward.json` first, then falls back
to `reward.txt`. Infrastructure failure must remain a Harbor error with no reward,
not a fabricated zero. Preserve `grade_result.json`, trial configuration, verifier
logs, image identity and per-stage timings. Oracle/no-op/invalid-repair verdicts
must match existing TrainDoctor checks before compatibility is called working.

## Trust boundary

Shared-container verification would let the agent affect scoring state. This task
explicitly uses separate verification. The grader consumes only the declared
source file. Harbor also transfers its agent publish directory; those additional
artifacts are ignored by this grader. The source file is untrusted: only a bounded regular
UTF-8 file and rejects links or unexpected submissions. A fresh image owns the
runtime, original fixture and scoring code. Agent checkpoint files, logs and
claimed correctness flags are never used as trusted grading records.

Both containers deny external networking and use read-only roots with explicit
scratch mounts. Agent commands run as uid 1000. The trusted verifier requires
root-owned scoring paths and drops worker privileges using existing launcher
machinery; capability/uid and trusted-ledger write-denial checks must pass before
executing submitted code. Harbor mounts verifier logs in the agent environment as
well: scoring must protect their root-owned directory and unlink stale reward
files before writing a new result, so prior open file handles cannot rewrite the
new reward inode. These are checks to execute, not a static security proof.
A container label alone does not justify marking an
unsandboxed run verified. The Linux bridge must fail closed when these checks are
unavailable. This task introduces no security guarantee from author-generated
attacks or static configuration alone.

## Observed local prerequisites

Read-only preflight found Docker client 29.5.3 and Docker.app, but the local
`desktop-linux` socket was absent and the server unavailable. Harbor was absent
from PATH and the prepared Python environment; the Apple `container` CLI was not
available. No daemon was started, package installed or container run by that
preflight. Thus no startup/verification overhead or Harbor verdict is asserted.

Before a model trial, prove oracle, no-op and completing-invalid controls locally,
measure memory/startup/verification overhead, and audit reward/ledger tampering.
Then use Harbor's external `BaseAgent` interface so MLX inference stays on the Mac
host while tools act through `BaseEnvironment`. Do not put MLX in the Linux image
or substitute hosted inference. This is a design path, not an implemented agent
integration or a local TRL/vLLM training solution.

Primary references:

- [Task overview](https://docs.harborframework.com/tasks/overview)
- [Separate verifier and artifact transfer](https://docs.harborframework.com/tasks/separate-verifier)
- [Offline lint](https://docs.harborframework.com/tasks/lint)
- [Run a job](https://docs.harborframework.com/jobs/run-a-job)
- [External agents](https://docs.harborframework.com/agents/custom-agents)
- [Pinned Docker provider](https://github.com/harbor-framework/harbor/blob/509b119db0d97b141dcd9c80f4f7ab861c47c2d6/src/harbor/environments/docker/docker.py)
- [Pinned task schema](https://github.com/harbor-framework/harbor/blob/509b119db0d97b141dcd9c80f4f7ab861c47c2d6/src/harbor/models/task/config.py)
