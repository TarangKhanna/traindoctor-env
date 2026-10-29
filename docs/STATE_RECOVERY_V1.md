# State only recovery demonstration

`traindoctor_env.state_recovery_v1` is one explicitly versioned engineering demonstration. It does not modify historical Task B, `Replay.outcomes()`, verifier v1.2, its scratch allowance, or historical results.

## Incident and observations

The existing correct MLP runs on two CPU/Gloo ranks for twelve optimizer steps, checkpointing every four. The launcher kills a worker after step 9; genuine step-4 and step-8 captures survive outside the job directory. An external recovery-artifact assembly then combines the step-8 model with either the step-4 sampler cursor or the step-4 optimizer momentum. The delivered bundle has a correctly recomputed manifest checksum. Integrity therefore cannot distinguish the two semantic inconsistencies. No training source is mutated.

`RecoveryIncident.status()` supplies the same initial worker-failure summary and action descriptions in both cases. `inspect_checkpoint()` exposes actual delivered tensor measurements and cursor position. `committed_history()` supplies measurements from genuine archived commits plus launcher-owned sample history. These are observations, without an injected-cause label, expected action or oracle verdict. The transparent `state_rule()` compares the delivered components with committed history; it is a sufficient-state baseline, not evidence of model reasoning.

## Recovery contract

The actual selected continuation is the object being graded. Each action starts from a copy of the same failed job and its authentic trusted ledger. Actions are fixed repository-authored operations: resume unchanged; restore only the sampler; restore only optimizer state; restore the complete committed bundle; resume the older coherent checkpoint; or restart from scratch. Manifest integrity is validated before launching every continuation. Original trusted captures are never rewritten or fabricated.

The data challenge is fixed across the paired comparison. Positions before step 8 use the original data salt; later positions use the same new salt in every action, reference and oracle. Thus committed pre-failure state remains compatible with the comparison. This deterministic demonstration challenge is not a claim of fresh adversarial protection against arbitrary code-writing agents.

Correctness is the existing sample, state-transition, rank, checkpoint and history contract, applied to the actual continuation using `reexec.check` with cost excluded. This is a new protocol, not historical Task B's fresh-source verification. The research oracle separately computes MLP forward/backward, rank averaging and SGD momentum using NumPy, conditional on genuine committed anchors and the shared data-service specification.

Cost is reported independently as raw redone optimizer steps, with no exemption. Scratch can be semantically correct while costly. Checkpoint continuity is not applicable to scratch. An illustrative scalar is zero for invalid outcomes and `max(0, 1-redone_steps/12)` for valid outcomes. That normalization is not a preregistered scientific metric or an established operational utility function. Job completion failures remain visible separately; a launch failure aborts the demonstration instead of being counted as semantic corruption.

The preemption and continuations require the existing macOS `sandbox-exec` backend; no silent unsandboxed fallback is allowed. The existing reference helper executes trusted repository code locally without OS isolation. This distinction is recorded in the companion report. The Seatbelt policy permits loopback communication and is not equivalent to the historical network-less Modal sandbox. This module accepts fixed actions, not arbitrary generated code.

## Reproduction

Use matching local environment and research checkouts containing this unreleased change. From the research checkout, with a prepared Python 3.12 environment containing the existing pinned CPU dependencies:

```bash
TD_KEY=state-recovery-demo-key PYTHONPATH=/path/to/traindoctor-env:. \
  uv run --no-project --offline /path/to/prepared/python \
  -m experiments.state_recovery_v1.demo --out results/a-new-run
```

The command refuses existing output paths. It executes the worker failure and every declared recovery, writes raw ledgers/captures, observations, authored-policy transcripts, independent oracle receipts and a result table. On a host where the OS job sandbox cannot run, it fails closed. Model downloads, inference, RL, Modal and API keys are unnecessary for this demonstration.

## Interpretation

The selective repairs can have opposite semantic outcomes. Full committed restoration remains legitimate and can be optimal in both states. This demonstration must therefore not be advertised as a task requiring conditional selection among all allowed actions, a difficult RL benchmark, or a model-agent result. Evidence ablation establishes that the initial summary cannot distinguish the selective targets; it does not make safe recovery impossible. Display-name perturbations are a structural noninterference check, not an execution-based population robustness result.
