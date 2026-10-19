# TrainDoctor's recovery verifier

**The objective is not to run an agent's patch safely; it is to determine whether that patch preserved the intended training computation.** TrainDoctor uses Modal's existing gVisor sandbox for isolation and implements the domain-specific execution record and state checks around it.

This verifier knows the expected behavior of an intentionally small PyTorch/Gloo data-parallel MLP, including its sampler, checkpoint schedule and SGD-with-momentum update rule. It is **not a general mathematical verifier** for arbitrary production PyTorch, FSDP, NCCL, TPU or JAX programs.

[Read the 2-minute example in the README](../README.md) · [Verifier implementation](../traindoctor_env/reexec.py)

## Trust boundary

![Agent-controlled ranks are isolated within Modal's sandbox; launcher-owned evidence is captured inside the sandbox and checked independently by a verifier outside it.](figures/fig_trust_boundary.svg)

| Component | What is and is not trusted |
| --- | --- |
| Agent patch and rank-written output | Untrusted. The agent can swallow errors, spoof metrics, write a misleading checkpoint or skip work. |
| Modal/gVisor | Isolates the modified job from the host and credentials; **does not judge correctness**. |
| Launcher and data server | Privileged **relative to agent-edited rank processes**, but run **inside** the sandbox. Their records are protected from normal rank writes. |
| Captured checkpoints | Trusted as snapshots of the bytes copied at a given moment; **their parameter values and metadata are not automatically correct**. |
| External verifier | Treats copied artifacts as data and independently checks sample accounting, reference states and expected optimizer transitions. |

The rank processes execute under an unprivileged UID; the launcher captures run history into protected storage. The verifier executes outside the sandbox. This split is an explicit attacker model, **not** an independently audited security guarantee.

## Controlled failure and process history

[`jobs/launcher.py`](../jobs/launcher.py) starts 3–4 ranks, sets deadlines and can inject a worker failure at a **step boundary**. The launcher parks ranks at the boundary, captures completed checkpoints, terminates the selected rank, and allows the prescribed restart. It records actual process exits, run identifiers, restart positions and repeated work. A patch that shortens the job or drops ranks should not earn credit for nominal success.

## Independent data delivery

[`jobs/dataserver.py`](../jobs/dataserver.py) serves rank-specific, salted batches over private socketpairs. Socket identity determines which rank requested data, rather than a field inside an untrusted JSON message. The server also records which sample IDs it served.

A served sample is **not necessarily a consumed training sample**: there may be prefetch, a crash or an abandoned step. The verifier combines the server's record with captured progress and independently reconstructed committed updates to detect skipped or duplicated *committed work*. Job-generated statements such as “all batches processed” are not proof.

## Checkpoint capture versus checkpoint validation

[`jobs/trusted.py`](../jobs/trusted.py) stores a launcher-owned ledger and copies checkpoint bytes when the launcher observes commit points. File paths are checked and captures are hashed. These records protect the *history* of what was written from later rank edits.

Checkpoint metadata or even an apparently correct final tensor can be fabricated. The verifier loads the captured state, checks model parameters, optimizer momentum and sampler position, and validates transitions across restarts. A hash of captured bytes identifies a copy; it does not establish the computation that created it was legitimate.

## Reconstructing the expected computation

[`traindoctor_env/reexec.py`](../traindoctor_env/reexec.py) checks the actual repaired job against the fault-free reference computation. For this fixed MLP/SGD setup it can also reconstruct expected parameter and optimizer updates from the batches the data server delivered, using a **pristine** implementation of the update rule. It checks the first committed update after restoration as well as later progress.

Main checks include consistent worker count and rank state, exact logical sample accounting, correct checkpoint/restored state, expected parameter transitions, bounded repeated work after failure, and integrity of non-editable files. Floating-point comparisons use numeric tolerances where appropriate; a legitimate repair is not required to produce bitwise-identical floating-point outputs.

Two operations often called “replay” are distinct here:

- **Repair re-execution:** run the patch in an isolated environment and capture what happened.
- **Independent update reconstruction:** recompute what the training algorithm should have done from the known reference rule and sample history.

## How the reward is defined

[`traindoctor_env/verifiers.py`](../traindoctor_env/verifiers.py) separates three success conditions. **V0** checks diagnosis; **V1** additionally checks supporting *pre-intervention* evidence citations; **V2** additionally requires sandboxed, correctness-verified repair and a non-destructive action. A forged log emitted after the agent edits the job is not valid evidence for the original diagnosis.

**V2 is full-task success, not repair selection alone.** In the Qwen3-4B comparison (reported in the research repository), the stronger-reward model cited better evidence but selected genuinely repairing actions at nearly the same rate as diagnosis-only GRPO. The training reward used precomputed exploratory repair outcomes; held-out scoring used live Modal re-execution.

## A concrete rejected shortcut

In the sandboxed A1_src scenario the incorrect patch `world_size: 4 → 1` (F11) eliminates the collective hang. The job finishes, but the verifier reports an incorrect world size, **72 missing and 8 duplicated samples**, and mismatched model state. The reference patch preserves the rank count and state and is accepted. The per-case table is in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research). The local quickstart's **96 missing samples** belong to a *different seed*.

Across v1, **86 incorrect-repair attempts were not accepted**: 48 were rejected by identified correctness invariants and 38 failed execution before useful artifacts. The 38 are **not evidence that an invariant caught the cheating strategy**. Controls: reference repairs 5/5, alternative legitimate repairs 9/10, no-op 0/5. The case tables and raw results are kept in the research repository.

## Known limitations

- **False rejection:** v1 imposes a checkpoint-format requirement that incorrectly rejects one behaviorally valid B_DUP restoration. A revised v1.1 check (`format_check='v1.1'`, opt-in) is implemented but was not used for the published v0.1 numbers.
- **Restricted task semantics:** the independent state reconstruction is specific to this task's known optimizer, sampler and model, not arbitrary distributed ML programs.
- **Limited attacks:** the red team was produced within the project and has no independent external security audit.
- **Limited learning signal:** current repair menus contain exactly one working fix, and the development tasks are too easy for strong coding agents. The agent comparison and a negative mutation pilot are in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research).
- **Reproducibility:** held-out seeds are withheld. Scored rollouts, live replay records, checksums and scripts are in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research), not in this repository; some executed Gloo test configurations have not been validated in an unrestricted public CI environment.

**Engineering takeaway:** sandbox isolation lets the agent's repair run; launcher-owned evidence and independent update checks make that repair **gradable for training correctness**.
