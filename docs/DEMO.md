# Demo: one failure, one repair, one fake repair

**This is an oracle demonstration.** The diagnosis and patch below come from the reference solver (`traindoctor_env/oracle.py`), not from a model. No agent transcript is included: no validated one exists in this repository.

Everything here is taken from recorded runs: the research repository's [`docs/RERUN_EXAMPLE.md`](https://github.com/TarangKhanna/traindoctor-research/blob/main/docs/RERUN_EXAMPLE.md) (seed 123, local), [`evidence/results/REDTEAM_TABLE.md`](https://github.com/TarangKhanna/traindoctor-research/blob/main/evidence/results/REDTEAM_TABLE.md) and [`evidence/results/quickstart_verified_replay.txt`](https://github.com/TarangKhanna/traindoctor-research/blob/main/evidence/results/quickstart_verified_replay.txt) (sandboxed). Reproduce the first with `python scripts/quickstart.py --fake F11` ([guide](QUICKSTART.md)).

## 1. Failure evidence

Task A1, seed 123, four ranks. All four ranks fail with `COLLECTIVE_TIMEOUT` at step 5. The per-rank collective trace at step 6 shows the difference:

```text
E3b6981 trace  [run0] rank 0 step=6 seq=12 all_reduce bucket=0 numel=80 pending
Ef4a4f6 trace  [run0] rank 1 step=6 seq=12 all_reduce bucket=0 numel=80 pending
E077217 trace  [run0] rank 2 step=6 seq=12 broadcast  bucket=0 numel=80 pending
E5d9a14 trace  [run0] rank 3 step=6 seq=12 all_reduce bucket=0 numel=80 pending
```

## 2. Diagnosis (oracle)

`COLLECTIVE_DIVERGENCE`, citing `E077217` and `E3b6981`: rank 2 enters `broadcast` where its peers enter `all_reduce`.

## 3. Patch (reference repair)

```diff
--- a/job/train.py
+++ b/job/train.py
@@ -77,10 +77,7 @@
             comm.broadcast(bucket, step, b)
             bucket.div_(comm.world)
         else:
-            if rank == 2 and step >= 6 and b == 0:
-                comm.broadcast(bucket, step, b)
-            else:
-                comm.all_reduce(bucket, step, b)
+            comm.all_reduce(bucket, step, b)
```

## 4. Replay and verifier output

Local replay of the same seed (not sandboxed, so the verifier forces `ok=False` for both patches and labels them INVALID; the invariant lines are what differ):

| patch | params match | samples match | state match | sample gaps | detail |
| --- | --- | --- | --- | --- | --- |
| gold | True | True | True | 0 | INVALID: unsandboxed re-execution |
| F11 (`world_size: 1`) | False | False | False | 96 | INVALID: unsandboxed re-execution; I6 world size differs from the reference |

Sandboxed verdicts (Modal gVisor, `exec_mode=modal_sandbox`). These come from different instances than seed 123, so counts are not comparable with the local table:

| instance | patch | accepted | first failing invariant |
| --- | --- | --- | --- |
| A1_src (red-team table) | gold | yes | none |
| A1_src (red-team table) | F11 | no | I6 world size; sample gaps 72, duplicates 8; parameters mismatch |
| A2 (quickstart replay, $0.008) | gold | yes | none |
| A2 (quickstart replay, $0.008) | F11 | no | I1 no_completed_relaunch |

The F11 patch removes the hang and the job completes, but the world size differs from the reference, samples are missing and the parameters are wrong. A seed-123 sandboxed replay is available through `python scripts/quickstart.py --modal` but has not been recorded.
