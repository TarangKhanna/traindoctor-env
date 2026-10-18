# TrainDoctor

An RL environment for debugging distributed PyTorch training jobs.

TrainDoctor gives a coding agent a failed multi-process PyTorch (Gloo) training job and asks it to diagnose and fix it. Each repair is re-run in a sandbox. A verifier outside the sandbox then checks whether training actually recovered: which samples each committed step consumed, whether checkpoint, optimizer and sampler state were restored correctly, and whether all ranks agree.

A job that exits cleanly is not necessarily repaired. In one task, rank 2 calls `broadcast` while ranks 0, 1 and 3 call `all_reduce`, and every rank times out. Setting `world_size` from 4 to 1 removes the hang and the job completes, but it is no longer the same training run: the verifier reports the wrong worker count, 72 missing and 8 duplicated samples, and parameters that differ from the reference ([per-case table](evidence/results/REDTEAM_TABLE.md)).

![Four ranks: rank 2 calls broadcast while ranks 0, 1 and 3 call all_reduce, so every rank times out with the same error](docs/figures/fig_failure.svg)

Research preview (v0.1): two synthetic failure families and exploratory results. [Quickstart](#quickstart) | [Results](#experiments-and-results) | [Limitations](#limitations)

## Quickstart

CPU only, Python 3.12 or 3.13, no GPU, API key or Modal account.

```bash
git clone https://github.com/TarangKhanna/traindoctor-env
cd traindoctor-env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/quickstart.py --fake F11 --out quickstart_out
```

This builds a failed four-process job, applies the reference repair and the `world_size: 1` fake, and runs the recovery checks on both (about 30 seconds after installation). The local run prints `INVALID: unsandboxed re-execution` by design: the verifier does not count results from agent-edited code that ran without isolation. The verified path uses a Modal Sandbox ([docs/QUICKSTART.md](docs/QUICKSTART.md), [recorded replay](evidence/results/quickstart_verified_replay.txt)).

## How recovery verification works

![An agent patch runs as untrusted ranks inside Modal's gVisor sandbox; a launcher-owned data server and checkpoint ledger record evidence; a verifier outside the sandbox checks it](docs/figures/fig_trust_boundary.svg)

Modal's gVisor sandbox isolates the agent's code but says nothing about whether the result is correct. That is the verifier's job.

Inside the sandbox, the launcher runs as root and the ranks as an unprivileged user. The launcher injects the worker failure, serves batches through a private-socket data server (sample identity comes from the socket, not from the job) and copies each checkpoint into its own ledger at commit time. The verifier runs outside and treats everything copied out as data. At each committed step it checks:

- each scheduled sample was consumed once;
- model, optimizer momentum and sampler position match the uninterrupted reference;
- where a repair changes the checkpoint layout, each transition between committed states matches an independent replay of the SGD-with-momentum update on the batches the data server delivered;
- lost work stays within one checkpoint interval, ranks agree, and protected files are untouched.

The reward has three levels: V0 (correct diagnosis), V1 (plus cited evidence that was observed), V2 (plus a verified repair and no destructive action). The checks are specific to this small MLP, sampler and optimizer, not a general verifier for production code. Details: [docs/VERIFICATION.md](docs/VERIFICATION.md), [ARCHITECTURE.md](ARCHITECTURE.md).

## Tasks

| Kind | Failure |
| --- | --- |
| A1 | A rank takes a different collective; all ranks time out |
| A2 | One rank stalls before communication; all ranks time out |
| B_SKIP, B_DUP | After a worker failure and restart, checkpoint and sampler disagree and samples are skipped or counted twice |
| B_OK | Healthy job; the correct answer is no fault |

See [docs/TASKS.md](docs/TASKS.md) and [docs/EXTENDING.md](docs/EXTENDING.md).

## Experiments and results

All results are exploratory. Raw results and checksums are in [`evidence/`](evidence/README.md).

**Training.** Qwen3-4B was fine-tuned with SFT and then GRPO, with a label-only reward (V0) or the recovery-aware reward (V2), six paired seeds each, and scored once on a sealed test set of 100 incidents. Success means V2 on the faulted task kinds: correct diagnosis, supporting evidence and a repair that passes live sandbox replay.

| Policy | Success |
| --- | ---: |
| Base model | 5.0% |
| SFT | 17.2% |
| Continued SFT, similar compute | 29.7% |
| GRPO, label-only reward (mean of 6 seeds) | 23.0% |
| GRPO, recovery-aware reward (mean of 6 seeds) | 33.4% |

The recovery-aware arm gained 10.36 percentage points over the label-only arm (95% bootstrap interval +5.83 to +15.26; positive in all six seeds). The gain came from evidence grounding, not better repair selection: both arms chose the recovering repair at the same rate (71.4% vs 71.5%) and label accuracy did not change. Continued SFT is not clearly different from the recovery-aware arm (+3.7 points; interval −1.7 to +9.1) and is a single run. Training used precomputed outcomes from a development container; test scores come from 369 live sandbox replays. Report: [evidence/results/TEST_REPORT.md](evidence/results/TEST_REPORT.md).

![Full-task success and working-repair selection for the two reward arms](docs/figures/fig_reward_outcome.svg)

**Attacking the verifier.** None of 86 fake repairs (swallow the exception, exit 0, shrink the world, raise timeouts, forge logs) was accepted: 48 were rejected by a correctness invariant and 38 failed during execution, which shows those cheats do not run, not that an invariant covers them. The reference repair was accepted 5/5 and alternative valid repairs 9/10 (one false rejection, fixed in a prepared v1.1 verifier). I wrote the attacks; there was no external review. See [REDTEAM_TABLE.md](evidence/results/REDTEAM_TABLE.md).

**Agent baselines.** On six development tasks, Claude Sonnet and Codex both produced patches that recovered all six; one Claude answer omitted a required evidence record (strict score 5/6 vs 6/6). The tasks are easy for strong agents and do not rank them ([report](experiments/cross_agent_dev6/BENCHMARK_REPORT.md)). A procedural-mutation pilot was stopped after Codex solved 24 of 24 attempts ([pilot](experiments/mutation_pilot/README.md)).

## Limitations

- Two synthetic task families; splits differ by seed, not by failure mechanism.
- Each repair menu has one recovering option, so V1 and V2 agree on 97.7% of development rollouts. The results do not show that recovery-based reward improves repair selection.
- Six paired seeds is a robustness minimum, not statistical power.
- Verified replay needs a Modal account. The macOS `sandbox-exec` profile is a development convenience, not a security boundary.
- Sealed test seeds are withheld (only their hash is published), so the test set cannot be regenerated here.
- No claim about production NCCL/FSDP failures or arbitrary training programs.

## Documentation

Guides are in [docs/](docs/README.md), the benchmark and pilot in [`experiments/`](experiments/README.md), raw results and protocols in [`evidence/`](evidence/README.md). Issues and pull requests are welcome, especially new red-team cases and tasks with several plausible repairs. Security reports: [SECURITY.md](SECURITY.md).

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Citation: [CITATION.cff](CITATION.cff).
