"""Offline stand-in for an executed Snapshot (no Gloo): records synthesized in the exact public format of
snapshot.collect()/public_records() for the job in jobs/template, from the scenario's own details (kill step, failing
step, ranks). Used only to size bundles against the token cap; the sample ids are random draws (1-2 digits, as the
real 96-sample data service yields). Calibration: the certificate-only floor for Task B at 4 ranks reproduces the Modal readiness run (2121-2433 est. vs
measured 2121-2432); Task A synthesizes ~1690-1740 vs the measured 2025-2041 (real A has more records than this model).
`pessimistic=True` lets every rank of the failed Task B run record samples through
the kill step inclusive, an upper bound on the real certificate (in a real run ranks stop at the collective that fails)."""
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from jobs.build import STEPS, CKPT_EVERY, materialize
from jobs.config import job_config
from traindoctor_env.scenario import Scenario, Truth, job_spec
from traindoctor_env.snapshot import SOURCE, evidence_ids, job_files, normalize

N_BUCKETS = 2


def synth_records(sc: Scenario, details: dict, cfg: dict, pessimistic: bool = True) -> list[dict]:
    rng = random.Random(f"synth:{sc.seed}")
    w, shard_steps = sc.world_size, cfg["data"]["shard_steps"]
    raw: list[tuple] = []

    first: dict = {}     # (rank, step) -> ids served in the failed run

    def ids(rk, s, ri):
        if ri == 0:
            first[rk, s] = [rng.randrange(96) for _ in range(cfg["batch_size"])]
            return first[rk, s]
        if sc.kind == "B_OK":                      # a correct restart replays the same logical samples
            return first.get((rk, s)) or [rng.randrange(96) for _ in range(cfg["batch_size"])]
        while True:                                # a faulty restart serves different ones (skipped / re-started position)
            v = [rng.randrange(96) for _ in range(cfg["batch_size"])]
            if v != first.get((rk, s)):
                return v

    def run_records(ri, start, end, fail=None, rank_fail=None, killed=False):
        """fail: step at which the run breaks (None = completes). Ranks log through the last step they finished."""
        run = f"run{ri + 1}"
        last = {r: (end - 1 if fail is None else fail - 1) for r in range(w)}
        status = "completed" if fail is None else "failed"
        err = "" if fail is None else " error=COLLECTIVE_TIMEOUT"
        raw.append(("job", -1, ri, last[0], 0, f"[{run}] resume_from={'ckpt-%d' % start if start else 'none'} started_at_step={start} -> {status} last_step={last[0]}{err}"))
        for rk in range(w):
            st = "exited" if fail is None else ("killed" if killed and rk == rank_fail else "failed")
            ex = "" if fail is None else (" exit=-9" if st == "killed" else " exit=1")
            raw.append(("process", rk, ri, last[rk], 0, f"[{run}] rank {rk} {st}{ex or ' exit=0'} last_step={last[rk]}"
                        + ("" if fail is None or st == "killed" else " COLLECTIVE_TIMEOUT")))
            i = 0
            raw.append(("log", rk, ri, -1, i, f"[{run}] rank {rk} INFO start world_size={w} resume_from={'ckpt-%d' % start if start else 'none'}"))
            top = last[rk] + (1 if pessimistic and killed and fail is not None else 0)
            for s in range(start, min(top, end - 1) + 1):
                i += 1
                if s % shard_steps == 0:
                    raw.append(("log", rk, ri, s, 2 * i, f"[{run}] rank {rk} INFO loaded shard {s // shard_steps} from data/part-{rk}/shard{s // shard_steps}.idx"))
                raw.append(("log", rk, ri, s, 2 * i + 1, f"[{run}] rank {rk} INFO step {s} loss={rng.random():.3f}"))
                raw.append(("samples", rk, ri, s, i, f"[{run}] rank {rk} step={s} samples={ids(rk, s, ri)}"))
                for b in range(N_BUCKETS):
                    raw.append(("trace", rk, ri, s, (s - start) * N_BUCKETS + b,
                                f"[{run}] rank {rk} step={s} seq={(s - start) * N_BUCKETS + b} all_reduce bucket={b} numel=96 done"))
                if (s + 1) % CKPT_EVERY == 0 or s + 1 == end:
                    raw.append(("log", rk, ri, s, 2 * i + 2, f"[{run}] rank {rk} INFO checkpoint ckpt-{s + 1} committed"))
    kind, r = sc.kind, details["rank"]
    ev = []
    if kind.startswith("B"):
        k = details["step"]
        s0 = (k // CKPT_EVERY) * CKPT_EVERY
        run_records(0, 0, STEPS, fail=k, rank_fail=r, killed=True)
        run_records(1, s0, STEPS)
        ev = [("submitted", "run1 resume_from=None"), ("failed", "run1 COLLECTIVE_TIMEOUT"),
              ("auto_restart", "run2 lost a worker; restarting from the latest checkpoint"),
              ("submitted", f"run2 resume_from=ckpt-{s0}"), ("completed", "run2")]
        ckpts = sorted({(s, ) for s in range(CKPT_EVERY, STEPS + 1, CKPT_EVERY)})
    else:
        f = details["step"]
        run_records(0, 0, STEPS, fail=f)
        seq = f * N_BUCKETS
        # the divergent / stalled rank, as the failing collective shows it
        raw[:] = [x for x in raw if not (x[0] == "trace" and x[3] == f and x[1] == r)]
        if kind == "A1":
            raw.append(("trace", r, 0, f, seq, f"[run1] rank {r} step={f} seq={seq} {details['bad_op']} bucket=0 numel=96 pending"))
        else:
            raw.append(("log", r, 0, f, 10**6, f"[run1] rank {r} INFO waiting for shard {f // shard_steps} at data/part-{w}/shard{f // shard_steps}.idx"))
        for q in range(w):
            if q != r:
                raw.append(("trace", q, 0, f, seq, f"[run1] rank {q} step={f} seq={seq} all_reduce bucket=0 numel=96 pending"))
        ev = [("submitted", "run1 resume_from=None"), ("failed", "run1 COLLECTIVE_TIMEOUT")]
        ckpts = [(s,) for s in range(CKPT_EVERY, f + 1, CKPT_EVERY)]
    for i, (e, d) in enumerate(ev):
        raw.append(("scheduler", -1, 0, i, 0, f"{e}: {d}"))
    for i, (s,) in enumerate(ckpts):
        raw.append(("ckpt", -1, 0, s, i, f"ckpt-{s} step={s} world_size={w}"))
    raw.sort(key=lambda x: x[:5])
    return [dict(id=i, kind=k, rank=rk, step=st, text=normalize(tx), source=SOURCE)
            for i, (k, rk, _, st, _, tx) in zip(evidence_ids(sc.seed, len(raw), 0), raw)]


def synth_snapshot(sc: Scenario, tmp: Path, pessimistic: bool = True):
    spec = job_spec(sc)
    materialize(tmp, spec.config, spec.bug)
    truth = Truth(sc.seed, sc.kind, sc.variant, sc.label, spec.task, [asdict(e) for e in spec.fix], spec.details)
    snap = SimpleNamespace(files=job_files(tmp), config=job_config(tmp), seed=sc.seed,
                           records=tuple(synth_records(sc, spec.details, spec.config, pessimistic)))
    return snap, truth
