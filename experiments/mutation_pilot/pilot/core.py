"""Shared plumbing of the mutation pilot: task table, snapshot execution of a mutant, fingerprints, live (Modal Sandbox)
replay with an on-disk cache and a spend guard. Imports the canonical modal-infra worktree read-only.

Task ids are neutral; the (seed, operator, variant) behind each id lives only in TASKS below and never reaches an agent
workspace (workspaces contain the buggy job files and the incident records only)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
SNAPS = OUT / "snaps"
MODAL_SITE = os.environ.get("TD_MODAL_SITE")   # optional: site-packages dir of an environment that has the modal client
CAP_USD = 2.20
LEDGER_PREFIX = "mutpilot-"
STEPS, CKPT_EVERY = 12, 4

OUT.mkdir(exist_ok=True)
if "TD_KEY" not in os.environ:
    raise RuntimeError("set TD_KEY (any random hex string) in the environment; it keys the data server and is never stored")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(HERE))
from generator.mutate import CANON, make_mutant, materialize  # noqa: E402  (also puts CANON on sys.path)

# neutral id -> (seed, operator, variant); order shuffled once so ids carry no operator signal
TASKS = {
    "p01": (40006, "O3", "b"), "p02": (40001, "O1", "b"), "p03": (40008, "O4", "b"), "p04": (40003, "O2", "a"),
    "p05": (40005, "O3", "a"), "p06": (40007, "O4", "a"), "p07": (40002, "O1", "a"), "p08": (40004, "O2", "b"),
}
# the incident each (operator, variant) must produce, as run statuses (validity filter of the generator)
EXPECT_STATUSES = {("O3", "b"): ["failed", "failed"]}
DEFAULT_B_STATUSES = ["failed", "completed"]


def mutant(tid: str):
    seed, op, v = TASKS[tid]
    return make_mutant(seed, op, v)


# ---------------------------------------------------------------------------------------------- snapshot
def build_snapshot(tid: str):
    """Execute the buggy job once (including the worker failure and automatic restart for task B), exactly as
    traindoctor_env.snapshot.execute does for catalogue scenarios, and freeze it. Local, trusted (repo-owned code)."""
    import shutil
    from dataclasses import asdict
    from jobs import manifest, trusted
    from jobs.config import job_config
    from jobs.dataserver import salt_for
    from jobs.resubmit import resubmit
    from traindoctor_env.scenario import Truth
    from traindoctor_env.snapshot import Snapshot, job_files, public_records, reference_run, run_names

    m = mutant(tid)
    out = SNAPS / str(m.seed)
    if (out / "truth.json").exists():
        return Snapshot.load(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    fx, spec, salt = out / "fixture", m.spec, salt_for(m.seed)
    materialize(fx, spec.config, spec.bug)
    if spec.task == "A":
        res = resubmit(fx, salt=salt)
        expect = res.status == "failed"
    else:
        res = resubmit(fx, kill=spec.kill, restart_on_failure=True, salt=salt)
        statuses = [json.loads(trusted.read_run_record(fx, r, "job.json"))["status"] for r in run_names(fx)]
        expect = statuses == EXPECT_STATUSES.get((m.op, m.variant), DEFAULT_B_STATUSES) and trusted.ledger(fx)[0]["kill_done"]
    if not expect:
        raise RuntimeError(f"{tid} ({m.kind}) did not play out as designed: {res.status} {res.error_class}")
    frontier = res.job.last_step + 1 if res.job else 0
    priv = fx / "_private"
    priv.mkdir()
    (priv / "reference.json").write_text(json.dumps(dict(reference_run(spec, salt), pristine_config=spec.config)))
    (priv / "meta.json").write_text(json.dumps(dict(frontier=frontier, kill=spec.kill, task=spec.task)))
    pub = out / "public"
    pub.mkdir()
    (pub / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in public_records(fx, m.seed)))
    (pub / "config.json").write_text(json.dumps(job_config(fx), indent=2, sort_keys=True) + "\n")
    (pub / "files.json").write_text(json.dumps(job_files(fx), indent=1, sort_keys=True) + "\n")
    # the harness view of the manifest: an unreadable (torn) file counts as "no checkpoints", as in jobs.manifest.load
    (pub / "manifest.json").write_text(json.dumps(manifest.load(fx), indent=2, sort_keys=True) + "\n")
    Truth(m.seed, m.kind, 0, f"{m.op}{m.variant}", spec.task, [asdict(e) for e in spec.fix], spec.details).save(out / "truth.json")
    (out / "mutant.json").write_text(json.dumps(m.describe(), indent=1, sort_keys=True))
    return Snapshot.load(out)


# ---------------------------------------------------------------------------------------------- fingerprint
TRACE_RX = re.compile(r"step=(\d+) seq=(\d+) (\w+) bucket=(\d+) numel=(\d+) (done|pending)")


def _first_divergence(snap) -> dict:
    """Where the incident first deviates from the clean reference, from harness-owned records only."""
    from jobs import trusted
    from traindoctor_env.snapshot import lineage
    task = snap.meta["task"]
    if task == "B":
        lin = lineage(snap.fixture_dir, STEPS)
        want = {int(r): {int(s): v for s, v in d.items()} for r, d in snap.reference["samples"].items()}
        for step in range(STEPS):
            for rank in sorted(want):
                if lin.get(rank, {}).get(step) != want[rank].get(step):
                    got = lin.get(rank, {}).get(step)
                    kind = "missing" if got is None else "different"
                    return dict(step=step, ranks=[rank], what=kind)
        return dict(step=None, ranks=[], what="none")
    calls: dict = {}
    for r in snap.records:
        if r["kind"] != "trace":
            continue
        mt = TRACE_RX.search(r["text"])
        if mt:
            step, seq, op, bucket, numel, st = mt.groups()
            calls.setdefault((int(step), int(seq)), {})[r["rank"]] = (op, int(bucket), int(numel), st)
    for key in sorted(calls):
        sigs = {rk: v[:3] for rk, v in calls[key].items()}
        if len(set(sigs.values())) > 1:
            maj = Counter(sigs.values()).most_common(1)[0][0]
            odd = sorted(rk for rk, v in sigs.items() if v != maj)
            diff = "+".join(n for n, a, b in zip(("op", "bucket", "numel"), sigs[odd[0]], maj) if a != b)
            return dict(step=key[0], ranks=odd, what=f"collective_signature_differs:{diff}")
    pend = sorted({(k[0]) for k, v in calls.items() if any(x[3] == "pending" for x in v.values())})
    return dict(step=pend[0] if pend else None, ranks=[], what="pending_without_signature_difference")


def _run_shape(snap) -> list:
    from jobs import trusted
    out = []
    for rec in trusted.ledger(snap.fixture_dir):
        job = json.loads(trusted.read_run_record(snap.fixture_dir, rec["run_id"], "job.json"))
        out.append(dict(status=rec["status"], error=job["error_class"], from_scratch=rec["start_step"] == 0,
                        resumed=rec["resume_from"] is not None))
    return out


def failing_invariants(o: dict | None) -> list:
    """Which recovery invariants a replay outcome violates (names only; counts reduced to > 0)."""
    if o is None:
        return ["no_outcome"]
    checks = dict(completed=o["ckpt_valid"], samples=o["samples_match_reference"], resumes=o["restores_consistent"],
                  transitions=o["transitions_consistent"], ranks=o["ranks_consistent"], params=o["params_match_reference"],
                  lost_work=o["lost_steps"] <= CKPT_EVERY, no_gaps=o["sample_gaps"] == 0, no_dups=o["sample_dups"] == 0,
                  state_equal=o["states_match_reference"])
    return sorted(k for k, ok in checks.items() if not ok)


def fingerprint(snap, noop_outcome: dict | None) -> dict:
    """Behavioural fingerprint of the BROKEN task. `mechanism` drops step/rank values (coarse); `full` keeps them."""
    div, shape = _first_divergence(snap), _run_shape(snap)
    mech = dict(invariants=failing_invariants(noop_outcome), what=div["what"], runs=shape, task=snap.meta["task"])
    full = dict(mech, first_step=div["step"], first_ranks=div["ranks"], world=snap.config["world_size"])
    h = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]  # noqa: E731
    return dict(mechanism=mech, mechanism_hash=h(mech), full=full, full_hash=h(full))


def dedup(fps: dict) -> dict:
    """fps: tid -> fingerprint. Tasks with equal fingerprints count as one."""
    by_mech, by_full = {}, {}
    for tid, f in fps.items():
        by_mech.setdefault(f["mechanism_hash"], []).append(tid)
        by_full.setdefault(f["full_hash"], []).append(tid)
    return dict(n_tasks=len(fps), n_unique_mechanism=len(by_mech), n_unique_full=len(by_full),
                mechanism_groups=sorted(by_mech.values()), full_groups=sorted(by_full.values()))


# ---------------------------------------------------------------------------------------------- live replay
def setup_modal() -> None:
    if MODAL_SITE and MODAL_SITE not in sys.path:
        sys.path.append(MODAL_SITE)


def spent_usd() -> float:
    from modal_train import budget
    rows = [json.loads(x) for x in budget.LEDGER.read_text().splitlines() if x.strip()]
    return sum(r["usd"] for r in rows if str(r.get("run_id", "")).startswith(LEDGER_PREFIX) and r["kind"] == "actual")


def outcome_dict(o) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.__dict__.items()}


def edits_key(edits) -> str:
    return hashlib.sha256(json.dumps([(e.path, e.old, e.new) for e in edits]).encode()).hexdigest()[:16]


def live_outcome(tid: str, snap, edits, tag: str = "") -> dict:
    """Trusted replay of `edits` over the buggy job in a network-less Modal Sandbox (verify_fix backend='modal').
    Cached per (task, edits). -> {status: ok|infra|invalid_edit, outcome|error, wall_s, cached}."""
    setup_modal()
    from modal_train import budget
    from traindoctor_env import reexec
    key = edits_key(edits)
    cache = OUT / "outcomes" / tid / f"{key}.json"
    if cache.exists():
        return dict(json.loads(cache.read_text()), cached=True)
    if spent_usd() > CAP_USD:
        return dict(status="infra", error="mutpilot Modal budget cap reached", wall_s=0, cached=False)
    t0, last = time.time(), None
    for attempt in range(2):
        try:
            rp = reexec.Replay(snap, allow_unsandboxed=False, backend="modal")
            try:
                for e in edits:
                    rp.edit(e)
                o = rp.outcome()
                info = rp.backend_info
            finally:
                rp.close()
            wall = time.time() - t0
            budget.record(f"{LEDGER_PREFIX}{tid}-{tag}-{key}-{int(t0)}", "mutpilot_verify", "cpu_per_hour", wall * 4, "actual")
            res = dict(status="ok", outcome=outcome_dict(o), wall_s=wall, backend_info={k: v for k, v in info.items() if k != "entry_stdout"},
                       cached=False)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(res, default=str))
            return res
        except ValueError as e:        # edit target missing / unsafe path: the repair itself is invalid, not infra
            return dict(status="invalid_edit", error=str(e), wall_s=time.time() - t0, cached=False)
        except BaseException as e:     # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:400]
            budget.record(f"{LEDGER_PREFIX}{tid}-fail-{int(t0)}-{attempt}", "mutpilot_verify_fail", "cpu_per_hour", (time.time() - t0) * 4, "actual")
    return dict(status="infra", error=last, wall_s=time.time() - t0, cached=False)


def verified(res: dict) -> bool:
    """Verified recovery: the trusted replay says ok, it ran under the Modal sandbox, nothing was tampered with."""
    if res.get("status") != "ok":
        return False
    o = res["outcome"]
    return bool(o["ok"] and o["sandboxed"] and o["exec_mode"] == "modal_sandbox" and not o["tampered"])
