# M0 §5.4-1: Executable fault proof (COLLECTIVE_MISMATCH on Gloo)

**Verdict: PASS.** A 4-rank Gloo job where rank 2 takes a rank-dependent branch into `broadcast` while ranks 0,1,3 `all_reduce` fails deterministically with a Gloo timeout on all 4 ranks; removing the branch (config flag) completes 20 steps, writes and reloads a checkpoint, with identical weights on all ranks.

## Versions / environment
macOS (Darwin 25.5.0), Apple M4, CPU only. Python 3.12 (uv venv, scratch), **torch 2.14.1** (PyPI wheel, `is_gloo_available() == True`, no numpy installed). `MASTER_ADDR=127.0.0.1`, free port per run, `torch.multiprocessing.start_processes(start_method="spawn")`, PG timeout 8 s, parent watchdog 60 s.

## Commands
```
PY=<scratch venv>/bin/python
$PY scripts/m0/gloo_fault_proof.py --mode fault --runs 5 --out scripts/m0/logs/fault_proof
$PY scripts/m0/gloo_fault_proof.py --mode fixed --runs 3 --out scripts/m0/logs/fault_proof
```
Artifacts (gitignored, `logs/`): `trace_rank*.jsonl`, `fr_rank*.json`, `result_rank*.json`, `ckpt.pt`, `ckpt_manifest.json`, `*_summary.json`.

## Fault: results (5 runs, same seed)
Fault: step 3, rank 2 calls `broadcast(grad_weight, src=0)`; others call `all_reduce(grad_weight)` (same shape [4,8]).

| run | ranks failed | error | t_fail per rank | last completed step | exit codes | wall |
|---|---|---|---|---|---|---|
| 0-4 | 0,1,2,3 (all) | `RuntimeError ... gloo/transport/uv/unbound_buffer.cc:65 Timed out waiting 8000ms for recv operation to complete` on ranks 0,1,2; `unbound_buffer.cc:101 ... send operation` on rank 3 | 8.4-9.1 s (ranks agree within 0.1 s inside a run) | 2 (all ranks) | 1,1,1,1 | 9.3-10.2 s |

Deterministic: yes. Exception type, message (modulo run-to-run time), rank-to-send/recv assignment, last step and exit codes identical across all 5; no watchdog kills. Failure is detected only after the full 8 s timeout, there is no early "mismatch" error, so the hang is a pure timeout.

Harness note: `mp.spawn`'s `ctx.join()` SIGTERMs siblings of the first failing rank, making exit codes nondeterministic (1 vs -15; seen in my first 5 runs). The script polls processes itself so every rank's own exit code is recorded.

## Per-rank trace (own JSONL; step 3 is the divergence)
```
rank0 {"step":2,"op":"all_reduce","seq":5,"shape":[4],"t_exit":...}
rank0 {"step":3,"op":"all_reduce","seq":6,"shape":[4,8],"error":"RuntimeError: ...Timed out waiting 8000ms for recv..."}
rank1 {"step":3,"op":"all_reduce","seq":6,"shape":[4,8],"error":"...recv..."}
rank2 {"step":2,"op":"all_reduce","seq":5,"shape":[4],"t_exit":...}
rank2 {"step":3,"op":"broadcast","seq":6,"shape":[4,8],"error":"RuntimeError: ...Timed out waiting 8000ms for recv..."}
rank3 {"step":3,"op":"all_reduce","seq":6,"shape":[4,8],"error":"...send operation..."}
```
The mismatch is identifiable from traces alone: all ranks agree on seq 0-5 and step; at seq 6 rank 2's `op` differs from the other three, and none has a `t_exit`. The raw exception text does not name the offending rank or op, so the trace is required.

## Flight recorder verdict
- **Partially yes.** The recorder is active on Gloo with `TORCH_FR_BUFFER_SIZE` (`TORCH_NCCL_TRACE_BUFFER_SIZE` is deprecated and warns). `torch._C._distributed_c10d._dump_fr_trace_json(True, False)` returns per-rank entries with `profiling_name` (`gloo:all_reduce` vs `gloo:broadcast` at `collective_seq_id` 7 on rank 2 only), sizes, dtypes, pg_id. Evidence: `fr_rank{0..3}.json`. The old `_dump_nccl_trace` does not exist in 2.14.1.
- **Limits on Gloo:** every entry stays `state: "scheduled"`, including completed collectives (no completed/retired info), no automatic dump on timeout (we dump from the exception handler), and `python -m torch.distributed.flight_recorder.fr_trace` fails on our dumps (`KeyError: ('0', 0)` in `build_collectives`, missing NCCL-style pg_config; `tabulate` not installed either).
- **Decision:** use our own JSONL trace as the primary source (has t_enter/t_exit/error); FR JSON is an optional extra, not relied upon.

## Fixed run (3 runs)
`rank_branch=False`: all 4 ranks `ok`, exit 0, last_step 19 (20 steps), ~0.5 s per rank, 1.5 s wall. Rank 0 wrote `ckpt.pt` + `ckpt_manifest.json`; reload checksum equals trained checksum; all 4 ranks have the identical weight sha256 prefix `447d5ecfe68c7cca` in all 3 runs. `FIXED_PASS: True`.

## macOS Gloo quirks
- `GLOO_SOCKET_IFNAME` not needed with `MASTER_ADDR=127.0.0.1` (default works). `lo0` also works. A bad name fails instantly (`Unable to find address for: <name>`), so a wrong interface is a cheap executable fault.
- IPv6: not needed; `lo0` has `::1` but 127.0.0.1 rendezvous works without it.
- `spawn` works; `fork` also worked in a 1-run fixed test (0.5 s), but spawn is the safe default (fork after torch thread init is unsafe in general).
- Port reuse: sequential runs each on a fresh ephemeral port (bind port 0, release, pass to rank 0) worked in 8 consecutive runs; not tested with a fixed reused port. A small race exists between releasing and rebinding the port.
- Startup is fast (~0.2 s init); torch emits a numpy-missing warning when numpy is absent (harmless).
- Faulty runs are slow only by the configured timeout (8 s), so ~10 s per scenario execution.

## Other fault reproducibility (§4 verify column, quick standalone check, not in repo)
| fault | reproduces deterministically? | evidence |
|---|---|---|
| `SHARDING_MISCONFIG` (ckpt shard shape vs new world size) | yes | `load_state_dict` RuntimeError "size mismatch for weight: copying a param with shape [2,4]... " twice, identical |
| `PREEMPTION_NO_CLEANUP` (stale lock blocks resubmit) | yes | `O_CREAT|O_EXCL` lock → `FileExistsError` 3/3 |
| `NONIDEMPOTENT_RESUBMIT` (double-append manifest) | yes | two submits → 2 identical manifest entries |
These are pure file-level logic, so determinism is by construction; M1 should implement them in the real job's resubmission script rather than inline.

## §5.4-1: PASS
