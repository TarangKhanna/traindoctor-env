# TrainDoctor

An execution environment for diagnosing and repairing distributed PyTorch training jobs. The verifier checks sample history, model parameters and optimizer state after recovery.

[Experiments and results](https://github.com/TarangKhanna/traindoctor-research) · [Architecture](ARCHITECTURE.md) · [Task suite](docs/TASKS.md)

## A job can finish and still be wrong

A two-rank CPU/Gloo job is interrupted after a checkpoint. Its delivered recovery bundle contains current model weights and momentum, but an older sampler cursor. Checksums are valid and the restart completes; samples repeat and the semantic verifier rejects it.

The [worked example](docs/STATE_RECOVERY_V1.md) also tests stale optimizer momentum, selective restoration, full restoration and recovery from scratch. Correctness and repeated work are reported separately. A NumPy continuation oracle in the research repository cross-checks the trajectories.

## Try it

Python 3.12:

```bash
git clone https://github.com/TarangKhanna/traindoctor-env
cd traindoctor-env
python3.12 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e '.[dev]'
python -m pytest -q tests/test_state_recovery_v1.py
```

To inspect the recorded recoveries, follow the companion repo's [saved-evidence quickstart](https://github.com/TarangKhanna/traindoctor-research#reproduce-the-saved-local-example). It needs no model or cloud account. [Fresh distributed execution](docs/STATE_RECOVERY_V1.md#reproduction) requires macOS `sandbox-exec`; the trusted reference helper runs locally without that sandbox.

## Implementation

| Component | Code and contract |
| --- | --- |
| Worker launch, data delivery and checkpoint capture | [jobs/](jobs/) |
| Interactive investigation, source edits and relaunch | [tools.py](traindoctor_env/tools.py) |
| Semantic re-execution checks | [reexec.py](traindoctor_env/reexec.py) · [Verification contract](docs/VERIFICATION.md) |
| Harbor task with a separate Docker verifier | [Task and setup](docs/HARBOR.md) |

Unit CI, fresh worker execution and sandboxed verification cover different boundaries. See [testing](docs/TESTING.md) and the [latest reproduction status](https://github.com/TarangKhanna/traindoctor-research/blob/main/docs/REPRODUCIBILITY.md).

## Scope

The workloads use a small MLP, CPU/Gloo and SGD with momentum. They do not establish production NCCL/FSDP coverage or a difficult agent benchmark: full restoration solves both recovery states. Historical SFT/GRPO experiments did not establish better repair selection, and autonomous local-model repair remains unproven. [Results and limitations](https://github.com/TarangKhanna/traindoctor-research#results).

Apache-2.0: [LICENSE](LICENSE) · [Citation](CITATION.cff) · [Security](SECURITY.md)
