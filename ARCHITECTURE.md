# Architecture

TrainDoctor diagnoses and verifies repairs to failed distributed training. **Modal/gVisor isolates execution; TrainDoctor verifies recovery.**

![Trust boundaries in TrainDoctor: untrusted rank processes, launcher-owned evidence inside Modal Sandbox, and independent verification outside.](docs/figures/fig_trust_boundary.svg)

The agent submits edits to permitted job files. Inside an isolated Modal Sandbox, the patched PyTorch ranks run as an unprivileged user, while a launcher and private-socket data server manage rank identity, failure injection, batch delivery, run history and checkpoint captures. A verifier **outside** the sandbox checks captured training state against a known-good reference. Copied checkpoints remain untrusted data until the independent checks pass.

## Code map

| Component | Implementation |
| --- | --- |
| Scenario definitions and observed evidence | [`traindoctor_env/scenario.py`](traindoctor_env/scenario.py), [`snapshot.py`](traindoctor_env/snapshot.py) |
| Agent tools and diagnosis/repair tasks | [`traindoctor_env/tools.py`](traindoctor_env/tools.py), [`tiera.py`](traindoctor_env/tiera.py) |
| Multi-rank training job | [`jobs/template/train.py`](jobs/template/train.py) |
| Fault injection, timeout and restart control | [`jobs/launcher.py`](jobs/launcher.py) |
| Trusted sample delivery and captured state | [`jobs/dataserver.py`](jobs/dataserver.py), [`trusted.py`](jobs/trusted.py) |
| Sandboxed execution integration | [`traindoctor_env/modal_backend.py`](traindoctor_env/modal_backend.py) |
| Reference state reconstruction and verification | [`traindoctor_env/reexec.py`](traindoctor_env/reexec.py) |
| Diagnosis/evidence/repair rewards | [`traindoctor_env/verifiers.py`](traindoctor_env/verifiers.py) |
| Tier A prompt, menu and answer scoring | [`traindoctor_env/bundle.py`](traindoctor_env/bundle.py), [`tiera.py`](traindoctor_env/tiera.py) |

**Re-execution** runs the modified job; **independent update reconstruction** calculates expected training state from the reference algorithm and delivered samples.

This preview covers synthetic PyTorch/Gloo training with two failure families, **not arbitrary production FSDP/NCCL workflows**. See [verifier mechanisms](docs/VERIFICATION.md), [task definitions](docs/TASKS.md) and [results](https://github.com/TarangKhanna/traindoctor-research).
