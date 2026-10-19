# TrainDoctor

An RL environment for debugging distributed PyTorch training jobs.

TrainDoctor gives a coding agent a failed multi-process PyTorch (Gloo) training job and asks it to diagnose and fix it. Each repair is re-run in a sandbox. A verifier outside the sandbox then checks whether training actually recovered: which samples each committed step consumed, whether checkpoint, optimizer and sampler state were restored correctly, and whether all ranks agree.

A job that exits cleanly is not necessarily repaired. In one task, rank 2 calls `broadcast` while ranks 0, 1 and 3 call `all_reduce`, and every rank times out with the same error.

![Four ranks at step 6: rank 2 calls broadcast while ranks 0, 1 and 3 call all_reduce, so all four ranks time out with COLLECTIVE_TIMEOUT](docs/figures/fig_failure.svg)

A tempting shortcut is to set `world_size` from 4 to 1 (fake repair F11). The hang disappears and the job completes, but it is no longer the same training run. In the quickstart scenario above (seed 123) the verifier reports the wrong worker count, 96 missing samples and parameters that differ from the reference. A separate red-team run on a different task instance (A1_src, replayed in a Modal sandbox) reported 72 missing and 8 duplicated samples for the same fake ([per-case table](https://github.com/TarangKhanna/traindoctor-research/blob/main/evidence/results/REDTEAM_TABLE.md)). The two sets of counts come from different seeds and are not comparable.

Preview (v0.1): two synthetic failure families. This repository is the environment and verifier; training and evaluation results are in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research). [Quickstart](#quickstart) | [Tier A](#tier-a-interface-and-rewards) | [Results](#results) | [Limitations](#limitations)

## Quickstart

CPU only, Python 3.12 or 3.13, no GPU, API key or Modal account.

```bash
git clone https://github.com/TarangKhanna/traindoctor-env
cd traindoctor-env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/quickstart.py --fake F11 --out quickstart_out
```

This builds a failed four-process job, applies the reference repair and the `world_size: 1` fake, and runs the recovery checks on both (about 30 seconds after installation). The local run prints `INVALID: unsandboxed re-execution` by design: the verifier does not count results from agent-edited code that ran without isolation. The verified path uses a Modal Sandbox ([docs/QUICKSTART.md](docs/QUICKSTART.md)). To use the package from another project: `pip install ".[modal]"` from a clone.

## How recovery verification works

![An agent patch runs as untrusted ranks inside Modal's gVisor sandbox; a launcher-owned data server and checkpoint ledger record evidence; a verifier outside the sandbox checks it](docs/figures/fig_trust_boundary.svg)

Modal's gVisor sandbox isolates the agent's code but says nothing about whether the result is correct. That is the verifier's job.

Inside the sandbox, the launcher runs as root and the ranks as an unprivileged user. The launcher injects the worker failure, serves batches through a private-socket data server (sample identity comes from the socket, not from the job) and copies each checkpoint into its own ledger at commit time. The verifier runs outside and treats everything copied out as data. At each committed step it checks:

- each scheduled sample was consumed once;
- model, optimizer momentum and sampler position match the uninterrupted reference;
- where a repair changes the checkpoint layout, each transition between committed states matches an independent replay of the SGD-with-momentum update on the batches the data server delivered;
- lost work stays within one checkpoint interval, ranks agree, and protected files are untouched.

The checks are specific to this small MLP, sampler and optimizer, not a general verifier for production code. Details: [docs/VERIFICATION.md](docs/VERIFICATION.md), [ARCHITECTURE.md](ARCHITECTURE.md).

## Tasks

| Kind | Failure |
| --- | --- |
| A1 | A rank takes a different collective; all ranks time out |
| A2 | One rank stalls before communication; all ranks time out |
| B_SKIP, B_DUP | After a worker failure and restart, checkpoint and sampler disagree and samples are skipped or counted twice |
| B_OK | Healthy job; the correct answer is no fault |

See [docs/TASKS.md](docs/TASKS.md) and [docs/EXTENDING.md](docs/EXTENDING.md).

## Tier A interface and rewards

Tier A is the single-turn form of the environment. `traindoctor_env.bundle.make_bundle` turns an executed snapshot into a compact text of evidence-id-tagged records plus a menu of repairs, capped at 3,072 Qwen3 tokens. `traindoctor_env.tiera` renders the chat prompt, parses the JSON answer (label, cited evidence ids, repair, whether it claims to have fixed the job) and scores it. Tier B is multi-turn: tools in `traindoctor_env/tools.py` expose the same snapshot and the agent edits the job's declared editable files, with the diff replay-verified.

The reward has three levels, each adding one requirement: V0 (correct diagnosis), V1 (plus cited evidence that was observed and is diagnostic for the mechanism), V2 (plus a repair that passes sandboxed re-execution, and no destructive action). A format violation scores zero. `audit/menu_leak.py` checks that the bundle and menu do not leak the label.

## Results

The environment was used to train and evaluate Qwen3-4B with a label-only reward (V0) and the recovery-aware reward (V2), and to attack the verifier. In the exploratory comparison on a held-out test set (its protocol was fixed before evaluation), the V2 arm scored 10.36 percentage points higher than the V0 arm; the gain came from evidence grounding, not from better repair selection. None of 86 fake repairs was accepted, though 38 of those failed during execution rather than being caught by an invariant. Two strong coding agents solved all six development tasks, so those tasks are easy and do not rank agents. These results are not reproduced by this repository. The scored rollouts, live-replay records, checksums and analysis scripts are in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research): the headline numbers there can be recounted from the stored raw files by script (recomputable), while the training runs, the strong-agent runs and the sandboxed live replays were executed once on specific hardware and a specific Modal account and are recorded rather than re-runnable from here (historical). This library provides the environment, verifier and red-team cases only; it contains no training or evaluation code, and held-out seeds are withheld.

## Limitations

- Two synthetic task families; splits differ by seed, not by failure mechanism.
- Each repair menu has one recovering option, so V1 and V2 agree on 97.7% of development rollouts. The results do not show that recovery-based reward improves repair selection.
- The training result rests on six paired seeds, a robustness minimum rather than statistical power, and is exploratory.
- Verified replay needs a Modal account. The macOS `sandbox-exec` profile is a development convenience, not a security boundary.
- No claim about production NCCL/FSDP failures or arbitrary training programs.

## Documentation

Guides are in [docs/](docs/README.md). Issues and pull requests are welcome, especially new red-team cases and tasks with several plausible repairs. Security reports: [SECURITY.md](SECURITY.md).

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Citation: [CITATION.cff](CITATION.cff).
