# TrainDoctor

**TrainDoctor: An executable RL environment for debugging and repairing distributed ML training failures.**

> **Research Preview v0.1.** Exploratory research code, two task families, no external review. Nothing here claims a new training algorithm or that RL improved repair selection (see Evidence).

An agent gets a failed multi-rank PyTorch/Gloo training job (logs, per-rank collective traces, config, source), proposes a repair, and a verifier decides whether training actually recovered with its computation intact. Failures are real CPU/Gloo processes, not canned text. Apache-2.0. Research code: results are exploratory (see [Limitations](#limitations)).

## The problem in one screen

A 4-rank Gloo job (seed 123, Task A1). Rank 2 takes a rank-dependent branch into a different collective, so its peers wait for an `all_reduce` that never completes and every rank dies with the same timeout. Output of `scripts/quickstart.py`, trimmed:

```
E004893 job       [run0] resume_from=none started_at_step=0 -> failed last_step=5 error=COLLECTIVE_TIMEOUT
Edde9ac process   [run0] rank 0 failed exit=1 last_step=5 COLLECTIVE_TIMEOUT      (ranks 1, 2, 3: identical)
E3b6981 trace     [run0] rank 0 step=6 seq=12 all_reduce bucket=0 numel=80 pending
Ef4a4f6 trace     [run0] rank 1 step=6 seq=12 all_reduce bucket=0 numel=80 pending
E077217 trace     [run0] rank 2 step=6 seq=12 broadcast bucket=0 numel=80 pending     <-- diverges
E5d9a14 trace     [run0] rank 3 step=6 seq=12 all_reduce bucket=0 numel=80 pending
```

Every rank reports the same error; only the collective trace shows who diverged. The repair an agent should find (and cite `E077217`, `E3b6981` for):

```diff
--- a/job/train.py
+++ b/job/train.py
-            if rank == 2 and step >= 6 and b == 0:
-                comm.broadcast(bucket, step, b)
-            else:
-                comm.all_reduce(bucket, step, b)
+            comm.all_reduce(bucket, step, b)
```

After the repair the job completes 12 steps and every invariant holds (`docs/quickstart_local_output.txt`).

## Why "the job exits 0" is not recovery

Many wrong repairs make the job stop failing. A fake repair that collapses the job to one process (`world_size: 1`, red-team case F11) also ends cleanly and completes its steps, yet the same quickstart run shows it broke the computation: `sample_gaps=96`, parameters and states differ from the uninterrupted reference. Other fakes in the red-team suite swallow the exception, raise timeouts, drop a rank, patch `torch.distributed`, or forge the job's own logs. The verifier therefore checks, at the committed optimizer-step boundary: samples consumed per step equal the reference schedule (none skipped, none in two committed updates), restored model/optimizer/sampler state equals what the checkpoint committed, lost work is within a checkpoint interval, all ranks end in the same state, and protected files were not tampered with.

## How a repair is verified

```
 agent diff (only declared editable files)
        |
        v
 +--------------------------------------------+        +----------------------------------+
 | disposable gVisor sandbox (Modal Sandbox)  |        | trusted verifier (outside)       |
 | no network, no credentials                 | -----> | recomputes every invariant from  |
 | pristine scenario + the agent's diff       | files  | copied-back checkpoints/ledger,  |
 | job re-runs (launch, kill, resume)         | as     | compares to uninterrupted        |
 +--------------------------------------------+ data   | reference; accepts or rejects    |
                                                       +----------------------------------+
```

Nothing the sandbox reports is trusted; sample ids come from a launcher-owned data server, not from logs the job writes. The strict verifier refuses to count an outcome that did not run in a sandbox (`exec_mode` must be `modal_sandbox`). See [Verifier and trust boundary](#verifier-invariants-and-trust-boundary).

## Quickstart

CPU only, no paid API, no Modal needed for steps 1-2. Python 3.12.

```bash
git clone https://github.com/TarangKhanna/traindoctor-env && cd traindoctor-env
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt      # torch 2.14.1, numpy, pydantic, pytest, hypothesis (pins equal uv.lock)
#   or, to import the packages elsewhere:  pip install ".[dev]"   /   pip install git+https://github.com/TarangKhanna/traindoctor-env
#   extras: ".[modal]" (Modal backend), ".[rl]" (MLX/RL stack, Apple Silicon). The scenario generator needs TD_KEY (any random hex) in the environment; quickstart.py sets one for you.
#   Linux: pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt

# 1-3: build the failed 4-rank job, show what the agent sees, apply the gold repair, re-run locally (about 30 s)
python scripts/quickstart.py --fake F11 --out quickstart_out
```

This runs a real 4-process Gloo job, prints the agent-visible records, the gold diff, and the local re-run with all invariants. The local re-run is labelled **local, UNVERIFIED (not sandboxed)**: the verifier forces `ok=False` and prints `INVALID: unsandboxed re-execution`, even though every invariant passes. That refusal is the intended behaviour, not a bug. Recorded output: [`docs/quickstart_local_output.txt`](docs/quickstart_local_output.txt).

**Verified path (optional, needs a Modal account, about $0.03):** replay the gold repair and the fake F11 repair in a Modal Sandbox.

```bash
pip install modal && modal token new
python scripts/quickstart.py --modal         # or directly:
TD_KEY=$(python -c "import secrets;print(secrets.token_hex(16))") \
  python scripts/redteam_suite.py --run quickstart --tasks A2 --cases gold,F11 --cap-usd 0.15
```

Expected: `gold accepted=True` with `exec_mode=modal_sandbox`, `F11 accepted=False` (rejected by invariant I1/I6). A recorded run is in [`docs/results/quickstart_verified_replay.txt`](docs/results/quickstart_verified_replay.txt). If the Modal client is installed in a separate environment, set `TD_MODAL_SITE` to its `site-packages`. The sandbox is network-less; the host process needs only the Modal API endpoints and `*.modal-storage.com` for downloads.

Unit tests:

```bash
python -m pytest -q --ignore=tests/test_first_update.py -m "not sweep and not executed"
python -m pytest -q --ignore=tests/test_first_update.py -m "executed and not sweep"   # real Gloo jobs, slower
```

See [Known test failures](#known-test-failures) for the exact current counts.

## Tasks

| kind | what happens | correct repair |
|---|---|---|
| A1 | same timeout on all ranks; a rank-dependent code path (or config entry) enters a different collective | remove the divergence |
| A2 | same timeout; one rank stalled before communication (data/compute path) and never enters the collective | fix the stall, not the timeout |
| B_SKIP | after a worker failure the job restarts and loss looks normal, but samples are skipped because checkpoint and sampler state disagree | make restore consistent |
| B_DUP | same, but samples are consumed twice | make restore consistent |
| B_OK | healthy job (control): the right answer is no fault | no repair |

**Tier A** is a single-turn bundle (capped at 3,072 Qwen3 tokens): the agent sees evidence-id-tagged records and a menu of repairs, and returns a label, cited evidence ids and a repair. It is the RL-tractable projection of Tier B, not the task. **Tier B** is multi-turn: the same snapshot exposed through tools plus the job's real source/config in a workspace, with real edits applied and re-executed. Faults are executed (CPU Gloo, 3-4 ranks) and every artifact carries `source: executed | simulated`.

## Verifier invariants and trust boundary

Verifier ladder: V0 label correct; V1 adds cited evidence that was observed and satisfies the mechanism's evidence predicate; V2 adds a verified repair by re-execution. Components are reported separately, and claimed-vs-verified is logged for every episode.

| `exec_mode` | what runs the repair | counts as verified? |
|---|---|---|
| `modal_sandbox` | disposable Modal Sandbox (gVisor, no network, no credentials); invariants recomputed outside it | yes: the intended path for untrusted repairs |
| `sandboxed` | local macOS `sandbox-exec` profile (`jobs/sandbox.py`); documented residual: loopback stays open for Gloo | development only |
| `dev_container` | trusted pinned code in local Docker | no: exploratory training reward only, never counted by strict V2 |
| unsandboxed local | no isolation | no: forced `ok=False` (INVALID) |

Eval and audit code must never reach the `dev_container` reward. Design rationale: [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md). Reporting security issues: [`SECURITY.md`](SECURITY.md).

## Evidence

All numbers below are exploratory and come from committed files under `docs/results/` (hashes in `docs/results/SHA256SUMS`; `docs/RELEASE_MANIFEST.md` maps each number to its file and command; sanitization of the public copies is described in `docs/results/SANITIZATION.md`).

**Verifier red-team (Modal Sandbox, 107 cases, every row `exec_mode=modal_sandbox`)**: [`docs/REDTEAM_TABLE.md`](docs/REDTEAM_TABLE.md)
- 86 attempted fake repairs, 0 accepted. 48 were rejected by an artifact invariant; 38 were rejected because execution failed before artifacts existed (no relaunch completed). **The 38 show that those cheats do not run; they do not show invariant coverage.** Invariant coverage is claimed for the 48 only.
- v1 controls: null patch 0/5 accepted; gold repairs 5/5; alternative valid repairs 9/10. The one false reject is B_DUP alt2 (sampler position recomputed in `restore()`), failing only a checkpoint-format check. Verifier v1.1, which accepts it and still accepts 0/85 verified cheats, is prepared but **not active** (`docs/VERIFIER_V11.md`).
- A borderline surface patch (A2: skip the shard wait) is accepted by v1 as outcome-equivalent; it motivated the A2v2 scenario (`docs/A2V2_SCENARIO.md`).

**RL study (Qwen3-4B LoRA, oracle SFT then GRPO; reward V2 = recovery-verified vs V0 = label only; 6 paired training seeds)**
- DEV split, exploratory: V2 arm minus V0 arm = +4.64 points, 95% CI [-0.69, +9.98]: inconclusive. Against compute-matched continued SFT (a single run) neither arm is a clear win. [`docs/results/SEEDS_3_5_REPORT.md`](docs/results/SEEDS_3_5_REPORT.md), [`docs/results/CONTROLS_REPORT.md`](docs/results/CONTROLS_REPORT.md)
- Sealed TEST (100 episodes, run once; V2 from 369 live Modal Sandbox replays): **+10.36 percentage points** (bootstrap 95% CI [+5.83, +15.26]) on strict evidence-grounded, replay-verified end-to-end success, 4 faulted kinds. Label accuracy did not change (-0.10), and **repair-selection accuracy did not change (71.4% vs 71.5%)**: there is no demonstrated improvement in choosing the recovering repair. The gain is entirely in citing the right evidence, because every menu has exactly one recovering repair, so V2 is V1 plus that repair. An independent recount matched. [`docs/results/TEST_REPORT.md`](docs/results/TEST_REPORT.md), [`docs/results/test/AUDIT_RECOUNT.txt`](docs/results/test/AUDIT_RECOUNT.txt)

Spend: about $8 of Modal credit for the whole v1 study ([`docs/results/LEDGER_ACTUAL_SUMMARY.txt`](docs/results/LEDGER_ACTUAL_SUMMARY.txt)).

## Limitations

- Two task families and one design; splits are by seed, not by mechanism. No claim about unseen failure mechanisms.
- Weak reward discrimination: each menu has one recovering repair, so V1 equals V2 on 97.7% of DEV rollouts. A recovery-verified reward can only add information if several plausible repairs differ in whether training recovers.
- The A2 wait-removal patch is accepted as outcome-equivalent.
- The raw v1 red-team JSON was overwritten by a later errored run and is lost; the v1 counts rest on the committed table (recount from the table).
- The red team was written by the project's own agents and reviews; there was no external red team. Per-class sample sizes are 2-5.
- RL controls (continued SFT) are single runs; training reward used the `dev_container` fix table, not independently verified recovery. The sealed TEST seed pool is not published (only the hash of the used seed list), so TEST cannot be regenerated from this repo.
- B_OK (healthy control) is solved about 0% because the prompt presupposes a fault; a neutral prompt exists but no model was retrained on it.
- Local sandboxing on macOS is a development convenience, not a security boundary; Linux/Modal executed tests have failures that were not individually isolated (see below).

## Known test failures

Run in a clean clone of this repository, fresh venv from `requirements-cpu.txt` (macOS arm64, Python 3.12, torch 2.14.1 CPU), 2026-10-19:

| selection | result |
|---|---|
| `pytest -m "not sweep and not executed"` (excluding `tests/test_first_update.py`, which needs `mlx`/Metal) | **211 passed, 32 skipped, 1 xfailed-by-design** (see below) |
| `pytest -m "executed and not sweep"` (real multi-process Gloo jobs) | **37 passed, 78 failed** |

- The 78 executed failures all raise `jobs.sandbox.SandboxUnavailable: job isolation was requested but sandbox-exec cannot be applied here`: these tests launch jobs under the macOS `sandbox-exec` profile, which cannot be nested inside the restricted environment used for this run. They are environmental, not known code defects, but **I did not re-run them on an unrestricted macOS machine, so that is inferred from the error text, not shown by a passing run.** Linux is not covered by the local-sandbox tests (the sandbox is macOS-only); on Linux they would fail the same way.
- `tests/test_modal_train.py::test_prompt_is_not_copied_into_modal_train` is a real, unfixed violation: `modal_train/synth.py` embeds the chat-template token. It is marked `xfail(strict=True)` rather than weakened.
- Skipped: 9 need `transformers` (token-cap tests), 8 need `sandbox-exec`, 14 test a sandbox-spec installer that is not part of this release, 1 needs a prompt cache. `tests/test_first_update.py` needs `mlx` (Metal) and is excluded.
- `-m sweep`, Docker DEV container, `modal_train` jobs and MLX tests were not run. CI (`.github/workflows/ci.yml`) runs only the first selection on Ubuntu; that selection has not been run on Linux by me.

## Related work and differentiation

Brief and non-exhaustive; I could only vouch for what the linked pages say.
- [AIOpsLab](https://arxiv.org/abs/2501.06706) and [Cloud-OpsBench](https://arxiv.org/abs/2603.00468): agent benchmarks for cloud/microservice operations (detection, localization, mitigation). TrainDoctor targets distributed ML training jobs and grades recovery with training-correctness invariants.
- [DDBench](https://arxiv.org/abs/2608.14863): agentic repair of 60 historical bugs in open-source distributed systems, graded by a hidden test oracle. TrainDoctor uses executable multi-rank ML-training failures and re-executes the fix to check training state.
- [SWE-Gym](https://arxiv.org/abs/2412.21139), [R2E-Gym](https://arxiv.org/abs/2504.07164), [SWE-RL](https://arxiv.org/abs/2502.18449), and SWE-bench-style environments: executable software-engineering tasks (issue fixing) for evaluation or RL, graded by unit tests. TrainDoctor's tasks are runtime failures of multi-process training, graded by recovery invariants.
- A Kubernetes SRE RL environment listed as "kube-sre-gym" / SRE-GYM (OpenEnv-compatible): I found only third-party listings, not a canonical page, so I make no comparison beyond noting it exists.
- [PyTorch Flight Recorder tutorial](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html): shows how to debug a stuck collective with trace analysis (NCCL). It is a debugging aid, not an agent environment. In our own check on Gloo/CPU the recorder logged ops but did not dump on timeout (`docs/m0/fault_proof.md`), so TrainDoctor writes its own per-rank collective trace.
- [Agentic Diagnosis for LLM Training at Scale (AMD)](https://rocm.blogs.amd.com/software-tools-optimization/maxtext-slurm-agentic-diagnosis/README.html): an agent workflow for diagnosing large training failures; a tool, not a benchmark with a recovery-checking verifier.
- RunSleuth: I could not find a source I could verify, so it is not described here.
- Reward-hacking and verifier-audit work such as [BenchJack](https://arxiv.org/abs/2605.12673) motivates auditing the reward with mutation tests and exploit probes, as done here.

TrainDoctor's combination: executable multi-rank ML-training failures, replay-verified recovery in a sandbox, and an audited reward. No claim of being first or only.

## Repository layout

| path | contents |
|---|---|
| `traindoctor_env/` | core environment: snapshot/replay, tools, Tier A bundle, verifiers, oracle (plain Python, no OpenEnv/Inspect/MLX) |
| `jobs/` | executable Gloo training job, fault templates, launcher, sandbox profile, data server, re-execution harness |
| `scripts/` | `quickstart.py`, red-team suite, Modal sandbox proof, TEST analysis/recount |
| `audit/` | menu-leak audit |
| `rl/`, `modal_train/` | prompt renderer, splits, SFT/GRPO helpers; Modal training/eval/sandbox backend (needs a Modal account) |
| `tests/` | pytest suite |
| `docs/` | `DESIGN_NOTES.md`, `results/` (raw results + `SHA256SUMS`), `prereg/`, `specs/`, `audits/`, `writeups/`, `RELEASE_MANIFEST.md` |
| `docker/dev`, `scripts/dev_container_run.sh` | optional Docker DEV path (training reward only) |

## Citation

```bibtex
@software{khanna2026traindoctor,
  author  = {Khanna, Tarang},
  title   = {TrainDoctor: An executable RL environment for debugging and repairing distributed ML training failures},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/TarangKhanna/traindoctor-env},
  license = {Apache-2.0}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).

## Contributing

Issues and pull requests are welcome, especially: new red-team cases (a repair the verifier accepts but should not, or rejects but should not), tasks where several plausible repairs differ in whether training recovers, and reproduction reports of the numbers above. Keep tests asserting a rule rather than current output; name snapshot tests `test_snapshot_*`. Security issues: see `SECURITY.md`.
