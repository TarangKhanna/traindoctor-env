# TrainDoctor

**A training job can finish successfully and still be wrong.**

TrainDoctor is an execution environment for investigating and repairing distributed PyTorch training jobs. It checks the recovered computation—samples, model parameters and optimizer state—rather than accepting a clean exit code.

This repository contains the runtime, investigation tools and verifier. [TrainDoctor Research](https://github.com/TarangKhanna/traindoctor-research) contains the experiments and execution evidence.

## A recovery that passes checksums and fails correctness

A two-rank CPU/Gloo job is interrupted after a checkpoint. Its training source is correct, but the delivered checkpoint combines components from different committed steps. The file checksum is valid. Training resumes and finishes. The recovered computation is wrong.

```text
Committed training state       Delivered recovery state
step 8: model + momentum       step 8: model + momentum
step 8: sampler cursor         step 4: sampler cursor
                                      │
                               restart completes
                                      │
                               samples repeat
                                      │
                               semantic check rejects
```

The [worked demonstration](docs/STATE_RECOVERY_V1.md) also tests the reverse problem: a current sampler with stale optimizer momentum. Selective restoration fixes one inconsistency while leaving the other intact. Full committed restoration fixes both; scratch recovery is valid but repeats more work. Correctness and recovery cost are reported separately.

The companion repository preserves the executed continuations and a NumPy implementation of the training updates. You can [recount the saved evidence](https://github.com/TarangKhanna/traindoctor-research#reproduce-the-saved-local-example) without a model or cloud account.

## What is implemented

- **Real failures:** multi-process Gloo communication failures and checkpoint/restart inconsistencies, with per-rank traces and launcher-owned execution records.
- **Investigation tools:** inspect logs, collectives, source, configuration, sample history and checkpoints. Tier B permits constrained source edits and relaunches.
- **Execution verification:** check sample accounting, optimizer transitions, checkpoint history, rank agreement and recovery cost against trusted records.
- **Explicit isolation:** historical evaluations used Modal/gVisor; the local recovery demo uses macOS `sandbox-exec`. Each records its backend. The local profile permits loopback and provides a different isolation boundary.

The code is organized around a small MLP, deterministic data delivery and SGD with momentum. It is an inspectable research environment, with no claim of coverage for arbitrary production training programs.

## Try it

The fastest path is the research repository’s [saved-evidence quickstart](https://github.com/TarangKhanna/traindoctor-research#reproduce-the-saved-local-example). It checks archived executions locally and reports which recoveries preserve training semantics.

To install this environment from source with Python 3.12:

```bash
git clone https://github.com/TarangKhanna/traindoctor-env
cd traindoctor-env
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q tests/test_state_recovery_v1.py
```

For a fresh local recovery demonstration, use the matching research checkout and the [execution commands](docs/STATE_RECOVERY_V1.md#reproduction). That path requires macOS and fails closed if its sandbox is unavailable. The older [quickstart](docs/QUICKSTART.md) demonstrates a communication fault; its unsandboxed local result is intentionally not accepted as verified.

## Harbor interoperability

One checkpoint code-repair task runs through Harbor with a separate semantic verifier. Executed controls accept the correct patch and reject no-op, completing-but-incorrect and forged-evidence controls; native TrainDoctor verdicts agree.

Follow the [pinned setup and control commands](docs/HARBOR.md), including the isolation requirements and documented lint exception. [Raw results and measured costs](https://github.com/TarangKhanna/traindoctor-research/blob/main/docs/HARBOR_VALIDATION.md) are available in the research repository. This is a validated interoperability smoke test, not a challenging agent benchmark.

## Agent and learning research

TrainDoctor exposes two interfaces: **Tier A**, a single-turn evidence bundle with a repair menu, and **Tier B**, interactive inspection and code editing. Historical Qwen3-4B SFT/GRPO experiments used Tier A. They did not establish improved repair selection or an advantage over continued SFT.

The newer [masked-loss study](https://github.com/TarangKhanna/traindoctor-research/blob/main/evidence/masked_mean_v1/REPORT.md) implements a working multi-turn local-model harness and a source-debugging task with hidden semantic regressions. Its published packet can be recounted without model inference. Real Qwen/Gemma episodes attempted investigation but did not apply a patch or relaunch; successful autonomous repair remains unproven. The harness and executable task exist independently of that negative model outcome.

The state-recovery demonstration establishes a concrete verification capability. It does not establish a difficult agent benchmark: a full restore works in both states. Checkpoint-transition replay also has established prior art. The detailed [research findings](https://github.com/TarangKhanna/traindoctor-research) preserve these boundaries and the negative results.

## Explore the implementation

| Start here | Purpose |
| --- | --- |
| [Recovery demonstration](docs/STATE_RECOVERY_V1.md) | Incident, actions, correctness contract and reproduction |
| [Architecture](ARCHITECTURE.md) | Runtime, evidence flow and trust boundaries |
| [Verifier](traindoctor_env/reexec.py) / [contract](docs/VERIFICATION.md) | Transition checks, invariants and known limits |
| [Tier B tools](traindoctor_env/tools.py) | Interactive investigation and editable workspace |
| [Launcher](jobs/) | Worker execution, data delivery and checkpoint capture |
| [Testing](docs/TESTING.md) | CPU tests, sandbox tests and historical coverage |
| [Detailed project record](docs/PROJECT_DETAIL.md) | Historical results, task shortcuts, verifier versions and caveats |

Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE). [Citation](CITATION.cff) · [Security](SECURITY.md).
