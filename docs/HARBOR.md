# Harbor compatibility: one checkpoint-repair task

One existing CPU/Gloo code-repair task now runs through native Harbor 0.25.0
(commit `509b119db0d97b141dcd9c80f4f7ab861c47c2d6`, schema 1.4) on the local Mac.
This proves bounded interoperability, not task difficulty, model improvement or
RL readiness. Original verifier v1.2 and historical evidence are unchanged.

## Executed controls

The initial native Harbor validation completed without infrastructure errors:

| Authored control | Harbor reward | Interpretation |
| --- | ---: | --- |
| Oracle | 1 | Correct checkpoint writer passes semantic verification |
| No-op (`nop`) | 0 | Original defect remains |
| Completing-invalid | 0 | Sampler persistence repaired; restored momentum discarded |
| Tamper | 0 | Author-generated reward/record tampering rejected |

These are scripted controls, not model-agent results or independent human review.
All four controls passed again against the final stale-`reward.txt` cleanup code.
Their task checksums agree, and the executed grader/core source hashes match the
saved source archive. Native Mac controls agree on every shared v1.2 verdict field.

Evidence, exact commands, timings and remaining limits are in the companion
research worktree: `docs/HARBOR_VALIDATION.md` and
`results/harbor_validation_20261030/`. These are local and unpublished.

## Run the controls

Keep this task inside the environment repository. Use a clean temporary working
directory, a local Docker daemon and the pinned Harbor installation; serialize
trials and retain sufficient host memory:

```bash
uv venv --python 3.12 /tmp/traindoctor-harbor
uv pip install --python /tmp/traindoctor-harbor/bin/python \
  'harbor @ git+https://github.com/harbor-framework/harbor.git@509b119db0d97b141dcd9c80f4f7ab861c47c2d6'
export PATH="/tmp/traindoctor-harbor/bin:$PATH"
# Start Docker Desktop locally, then use a clean directory without dotenv files.
mkdir -p /tmp/traindoctor-harbor-cwd
cd /tmp/traindoctor-harbor-cwd
TASK=/absolute/path/to/traindoctor-env/harbor/tasks/training-repair-01
export PYTHON_DOTENV_DISABLED=1 HARBOR_TELEMETRY=0
harbor lint "$TASK" --json
harbor lint "$TASK" --json --ignore compose-additional-capabilities
harbor run -p "$TASK" -a oracle -e docker -n 1 -k 1 -r 0 --force-build
harbor run -p "$TASK" -a nop -e docker -n 1 -k 1 -r 0
```

Default lint reports the explicit root-verifier capabilities; that finding is
preserved in the receipts. The reviewed invocation ignores only
`compose-additional-capabilities`, rather than suppressing other lint checks.
Root needs those capabilities for preemption and privilege dropping; workers must
have zero capabilities. `harbor check` invokes an LLM rubric and is outside scope.
Do not pass dotenv files, API secrets or host credential forwarding options.

## Runtime and trust boundary

The agent edits `/submission/ckpt.py` as uid 1000. A fresh, separate verifier
consumes only that bounded regular source file, never agent-authored checkpoint
records or claimed verdicts. Harbor also transfers its publish directory; the
grader ignores those additional artifacts. Trusted fixture, launcher and scoring
code remain root-owned; submitted rank code runs with dropped UID/capabilities.

Harbor's provider policy is declared `public` because this Docker backend rejected
its `no-network` capability request. **Actual networking is denied by both custom
Compose files' `network_mode: none`**. The worker proof checks that only `lo` is UP,
allows dormant kernel interfaces, requires no `NET_ADMIN` capability and verifies
an immediate external routing/policy denial. This is not Harbor policy enforcement
or a general security guarantee from author-generated attacks.

Both roots remain read-only, with PID limits and explicit writable storage.
Harbor exposes verifier-log paths during agent operation, so the grader protects
root-owned output paths and removes stale reward files before writing a fresh
result. Both `reward.json` and `reward.txt` matter: Harbor prefers JSON, then falls
back to text. Infrastructure failure must produce an error with no reward.

Docker Desktop demonstrated a transport mismatch: `docker exec` could read a
regular file inside tmpfs while `docker cp` reported it missing. Transported
`/submission`, `/logs` and the agent's `/solution` therefore use project-scoped
named volumes, with no global names. `/tmp` and verifier `/work` remain bounded
tmpfs. Initial directory ownership comes from the images; scoring paths stay
root-protected.

## Reuse and scope

The task lives in `harbor/tasks/training-repair-01/`: standard `task.toml`,
`instruction.md`, environment, tests and oracle solution. It reuses existing
runtime/verifier code without vendoring a second runtime or building an adapter
framework. Native Compose builds from the repository root; Dockerfile-specific
allowlists exclude secrets, Git data, results and caches. Force rebuilds after
runtime source changes outside the task directory. Existing exact Torch pins use
only the official CPU-wheel index, with no CUDA/version fallback.

The task retains the historical fresh-source verification objective: test whether
the submitted code makes a subsequent preemption/restart correct. The agent's own
diagnostic execution is not the final scored continuation. The source-mutation
shortcut remains, so this is an integration smoke test, not a discriminative
learning benchmark. No Harbor model trial or SFT/RL result is established here.
A future MLX integration would keep inference on the Mac host through Harbor's
external `BaseAgent` interface; it is not implemented by this packaging work.

Primary references: [separate verification](https://docs.harborframework.com/tasks/separate-verifier),
[lint](https://docs.harborframework.com/tasks/lint),
[external agents](https://docs.harborframework.com/agents/custom-agents),
[pinned Docker provider](https://github.com/harbor-framework/harbor/blob/509b119db0d97b141dcd9c80f4f7ab861c47c2d6/src/harbor/environments/docker/docker.py).
