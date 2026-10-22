# TrainDoctor - Verifiable RL Environment for Distributed PyTorch Training Recovery

TrainDoctor is an executable RL environment (RLVR) in which an agent repairs a failed multi-process PyTorch training job. The difficulty is that a repair can make the job finish while silently breaking training: setting `world_size: 1` removes a collective hang and the job completes, but in one red-team case 72 scheduled samples were never trained on and 8 were trained on twice.

So a finished job is not taken as success. Each candidate repair is re-executed in a network-less Modal (gVisor) sandbox, and a verifier outside the sandbox checks it against records written by the launcher rather than by the job:

- sample delivery: which samples the launcher-owned data server actually served, against an uninterrupted reference run;
- checkpoint history: every checkpoint copied by the launcher at commit time, so later edits and fabricated older checkpoints are detected;
- optimizer updates: committed model and momentum state must equal the reference, or, where a repair changes the checkpoint layout, each committed transition must match an independent replay of the SGD-with-momentum update on the delivered batches.

[Quickstart](#quickstart) · [Demo](docs/DEMO.md) · [Task suite](docs/TASKS.md) · [Architecture](ARCHITECTURE.md) · [Research results](https://github.com/TarangKhanna/traindoctor-research)

Agents investigate executed 3- or 4-process (Gloo) failures, name the cause, cite evidence and repair the job. The suite has four fault kinds in two families, plus a healthy-job control:

- Communication failures: a rank enters a mismatched collective (A1), or a rank stalls before communication and never reaches the collective (A2). In both, the ranks that report a failure show the same collective timeout; in A2 the stalled rank itself is killed by the launcher watchdog.
- Silent corruption after recovery: a worker fails, the job restarts, loss looks normal, but checkpoint and sampler state disagree, so samples are skipped (B_SKIP) or processed twice (B_DUP).
- Healthy control: a job with no fault (B_OK), where the correct answer is no fault.

Tier A provides a single-turn evidence bundle and repair menu for RL training and evaluation. Tier B exposes the same snapshot through interactive tools for logs, traces, source and config, with code edits applied and re-executed. Tier B is a development environment, not a validated multi-turn RL benchmark.

Repairs are checked against launcher-controlled evidence and an independent reconstruction of optimizer state ([how it works](#how-recovery-verification-works)).

## Worked example: a repair that runs but is wrong

In the A1 task, rank 2 calls `broadcast` while ranks 0, 1 and 3 call `all_reduce`, and every rank times out with the same error.

![Four ranks at step 6: rank 2 calls broadcast while ranks 0, 1 and 3 call all_reduce, so all four ranks time out with COLLECTIVE_TIMEOUT](docs/figures/fig_failure.svg)

Setting `world_size` from 4 to 1 (fake repair F11) removes the hang and completes the job while changing the training run.

- Quickstart scenario (seed 123): the verifier reports the wrong worker count, 96 missing samples and parameters that differ from the reference.
- Red-team run (task instance A1_src, replayed in a Modal sandbox): 72 missing and 8 duplicated samples for the same fake ([per-case table](https://github.com/TarangKhanna/traindoctor-research/blob/main/evidence/results/REDTEAM_TABLE.md)). The two sets of counts come from different seeds and are not comparable.

## Where to look

- Verifier: [`traindoctor_env/reexec.py`](traindoctor_env/reexec.py)
- Tier A (bundle, prompt, scoring): [`bundle.py`](traindoctor_env/bundle.py), [`tiera.py`](traindoctor_env/tiera.py)
- Tier B (tools): [`tools.py`](traindoctor_env/tools.py)
- Launcher, data server, checkpoint capture: [`jobs/`](jobs/)
- Design: [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/VERIFICATION.md](docs/VERIFICATION.md)

Research preview (v0.1): executable CPU/Gloo workloads with two synthetic failure families. It is not a general-purpose verifier for arbitrary production training jobs. This repository holds the environment and verifier; training and evaluation results are in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research).

## Quickstart

CPU only, Python 3.12 or 3.13, no GPU, API key or Modal account.

```bash
git clone https://github.com/TarangKhanna/traindoctor-env
cd traindoctor-env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/quickstart.py --fake F11 --out quickstart_out
```

This builds a failed four-process job and checks the reference repair and the `world_size: 1` fake (about 30 seconds after installation). The local run prints `INVALID: unsandboxed re-execution` by design: the verifier does not count results from agent-edited code that ran without isolation. The verified path uses a Modal Sandbox ([docs/QUICKSTART.md](docs/QUICKSTART.md)). To use the package from another project: `pip install ".[modal]"` from a clone.

## How recovery verification works

![An agent patch runs as untrusted ranks inside Modal's gVisor sandbox; a launcher-owned data server and checkpoint ledger record evidence; a verifier outside the sandbox checks it](docs/figures/fig_trust_boundary.svg)

Modal's network-less gVisor sandbox isolates agent-edited code; the external verifier checks recovery.

Inside the sandbox, the launcher runs as root and the ranks as an unprivileged user. The launcher injects the worker failure, serves batches through a private-socket data server that determines sample identity independently of job-written logs and copies each checkpoint into its own ledger at commit time. The verifier runs outside and treats everything copied out as data.

- Trusted (repository code the agent cannot edit): the launcher, data server, checkpoint capture and ledger, the reference run and the verifier. The launcher shares the sandbox with the ranks and is separated from them by user id (root versus an unprivileged user), not by a separate machine.
- Untrusted, checked against the trusted records: the agent's edits and everything the ranks write (logs, metrics, checkpoint files, exit codes).

At each committed step the verifier checks:

- each scheduled sample was consumed once;
- model, optimizer momentum and sampler position match the uninterrupted reference;
- where a repair changes the checkpoint layout, each transition between committed states matches an independent replay of the SGD-with-momentum update on the batches the data server delivered;
- lost work stays within one checkpoint interval, ranks agree, and protected files are untouched.

The checks are specific to this small MLP, sampler and optimizer, not a general verifier for production code. Details: [docs/VERIFICATION.md](docs/VERIFICATION.md), [ARCHITECTURE.md](ARCHITECTURE.md).

## Task evidence

The ranks that report A1 and A2 failures show the same timeout, but the Tier A process records show a watchdog-killed rank only in A2, so process status alone separates A2 from A1 (a post-hoc audit on 120 new development seeds, [report](results/status_audit/run1/REPORT.md)). It cannot separate A1_src from A1_cfg. See [docs/TASKS.md](docs/TASKS.md) and [docs/EXTENDING.md](docs/EXTENDING.md).

## Tier A interface and rewards

`traindoctor_env.bundle.make_bundle` builds evidence-id-tagged records and a repair menu, capped at 3,072 Qwen3 tokens. `traindoctor_env.tiera` renders the chat prompt, parses the JSON answer (label, cited evidence ids, repair, claimed recovery) and scores it. Tier B tools in `traindoctor_env/tools.py` permit edits to declared editable files, with the diff replay-verified.

The reward has three levels, each adding one requirement: V0 (correct diagnosis), V1 (plus cited evidence that was observed and is diagnostic for the mechanism), V2 (plus a repair that passes sandboxed re-execution, and no destructive action). A format violation scores zero. `audit/menu_leak.py` checks the bundle and menu for label leakage; it does not establish that none exists (see Limitations).

## Verifier v1.2 (default) and optional menu v3

`format_check='v1.2'` adds `history_intact` to v1.1: older checkpoint entries must be byte-identical to an earlier launcher capture, rejecting fabricated older checkpoints. v1.2 is the default for new verifications; the published results used v1 and reproduce with `format_check="v1"`. See [docs/VERIFICATION.md](docs/VERIFICATION.md).

Menu v3 is opt-in (`make_menu(..., version="v3")`), with valid alternatives and repairs that complete while violating training semantics. Its reconciled Modal-sandboxed audit covers 720 candidates: 60 reference repairs, 115 valid alternatives, 271 incorrect-but-completed repairs, 274 non-completions and 0 infrastructure errors. v1.2 accepted all 175 repairs classified valid and rejected all 271 completed-but-invalid ones under the suite's reference-based criteria; v1 has 12 false rejections. Classification and verification share execution records and computed fields, so this is a consistency check, not an independent false-acceptance rate, and an author-designed development outcome rather than external validation. The experiment is NO-GO for RL. Sources: [candidate accounting](https://github.com/TarangKhanna/traindoctor-research/blob/main/experiments/menu_v3_dev/ACCOUNTING.md), [report](https://github.com/TarangKhanna/traindoctor-research/blob/main/experiments/menu_v3_dev/REPORT.md).

## Results (historical v0.1)

Qwen3-4B was trained (LoRA SFT, then GRPO) on Modal cloud GPUs and evaluated in this environment, and the verifier was attacked. Claude Sonnet and Codex were evaluated separately as existing coding agents; they were not trained here. All results are exploratory and are not reproduced by this repository, which has no training or evaluation code and withholds the held-out seeds. Statistics, uncertainty and provenance are in the research repository's [evidence map](https://github.com/TarangKhanna/traindoctor-research/blob/main/docs/EVIDENCE_MAP.md) and [erratum](https://github.com/TarangKhanna/traindoctor-research/blob/main/docs/ERRATUM_mmd.md).

- Reward comparison: on a held-out evaluation set, the V2 arm (recovery-aware reward) scored 10.36 points above the V0 arm (diagnosis-only reward); the observed gain accompanies improved evidence citation, while repair-choice rates were similar (71.4% vs 71.5%).
- Against continued SFT: a single run (compute comparability was not established) scored 29.7%; the V2 arm's +3.7-point difference is not clearly nonzero (95% interval -1.7 to +9.1).
- Development split: V2 minus V0 was +4.64 points (interval -0.69 to +9.98), which is inconclusive; the minimum meaningful difference (7.2 in the frozen protocol, 20 in its amendment) was never reconciled.
- Verifier attacks: none of 86 fake repairs was accepted, though 38 failed during execution rather than being caught by an invariant, and the attacks had no external review.
- Coding agents: Claude Sonnet and Codex, evaluated separately from the Qwen3-4B results, both recovered all six development tasks (strict scores: Claude 5/6 for one missing evidence record, Codex 6/6), so the tasks are easy and do not rank agents.

## Limitations

- The displayed job seed in `config.yaml` can be inverted to the scenario seed, and hence the label, by running the public generator over a known seed range (about 94% of 400 sampled scenarios in an owner-side check). Tier A models cannot execute code, but tool-using agents could; the leak is unresolved, and historical results are unchanged.
- Two synthetic task families; splits differ by seed, not by failure mechanism.
- In the historical v0.1 experiment, each repair menu has one recovering option, so V1 and V2 agree on 97.7% of development rollouts. The results do not show that recovery-based reward improves repair selection.
- The training result rests on six paired seeds, the minimum for a consistency check rather than statistical power, and is exploratory.
- Verified replay needs a Modal account. The macOS `sandbox-exec` profile is a development convenience, not a security boundary.
- Process-status records in the Tier A bundle identify A2 against A1 perfectly (40 of 40 each) but not A1_src against A1_cfg, so a model can use that shortcut instead of the traces. The environment was not changed.
- Tier B has not been validated as a multi-turn RL benchmark.
- No claim about production NCCL/FSDP failures or arbitrary training programs.

## Documentation

For a 5-minute overview, read this README and [ARCHITECTURE.md](ARCHITECTURE.md).

| Guide | Contents |
| --- | --- |
| [QUICKSTART.md](docs/QUICKSTART.md) | Install, local and sandboxed quickstart, unit-test commands |
| [DEMO.md](docs/DEMO.md) | Oracle evidence, diagnosis, patch, replay, gold and fake verdicts |
| [TASKS.md](docs/TASKS.md) | Five task kinds and two tiers |
| [VERIFICATION.md](docs/VERIFICATION.md) | Invariants, `exec_mode` trust distinctions, red-team details |
| [EXTENDING.md](docs/EXTENDING.md) | Package use and task extensions |
| [TESTING.md](docs/TESTING.md) | Test selections, historical counts, known skips and Modal regression |
| [TIERB_READINESS.md](docs/TIERB_READINESS.md) | Multi-turn RL readiness findings and proposed work |
| [figures/](docs/figures/) | `fig_failure` (`world_size:1` fake removes a hang) and `fig_trust_boundary` |

Contributions: new red-team cases and tasks with several plausible repairs. Security reports: [SECURITY.md](SECURITY.md).

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Citation: [CITATION.cff](CITATION.cff).
