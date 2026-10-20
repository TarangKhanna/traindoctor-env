"""Fix verification by re-execution. A Replay is the agent's workspace: a working copy of the snapshot's fixture
(never its _private material) whose job/ files can be edited and from which the job can be relaunched through
jobs.resubmit. The verdict never trusts that workspace's runs: outcome() re-executes the final job/ via the
standard recovery protocol (Task A: relaunch from the latest valid checkpoint of the incident; Task B: a fresh run
with the incident's worker failure and the scheduler's automatic restart) and judges it from what the launcher
itself recorded (jobs/trusted.py): checkpoints copied when committed, the data server's log, process history.

Grading. FAST PATH (sufficient): every committed checkpoint (model, optimizer momentum, sampler position) equals
the uninterrupted reference's state at the same step within TOLERANCE, and the run completed with the final
checkpoint committed. FALLBACK (a valid repair need not be state-equal, e.g. a different checkpoint layout): the
independently specified invariants (checkpoint format and sampler position, each transition between consecutive
committed states reproduced by the pristine update rule on the logged samples (this is restored-state
consistency at each resume), and exact logical sample accounting). Either way: lost work over every recovery attempt
<= one checkpoint interval, ranks agree, no tampering. Everything runs in this (the launcher's) process, outside
the sandbox the job runs in."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import logging
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from jobs import manifest, safefs, sandbox, trusted
from jobs.build import EDITABLE, Edit, apply_edits
from jobs.dataserver import SaltSchedule, batch_tensors, salt_at
from jobs.resubmit import resubmit
from traindoctor_env.snapshot import Snapshot, lineage, reference_for_config, run_names, step_batches

TOLERANCE = 1e-5          # max abs difference between two numerically different but equivalent states
FAST_TOL = TOLERANCE      # state-equality tolerance of the fast path
# Repo-owned code only (CI): run verification without OS isolation. Never for agent-edited code on a dev machine.
def ci_unsandboxed_allowed(env=None, platform: str | None = None) -> bool:
    """TD_CI_UNSANDBOXED=1 is honored only on a Linux CI runner (CI set). On a developer machine it is ignored."""
    env = os.environ if env is None else env
    return (sys.platform if platform is None else platform) == "linux" and bool(env.get("CI")) \
        and env.get("TD_CI_UNSANDBOXED") == "1"


ALLOW_UNSANDBOXED = ci_unsandboxed_allowed()
DEV_CONTAINER_MODE = False   # training-only, set by the training harness; never set in eval/audit paths
B_VALIDATION_SCRATCH_RUNS = 1   # Task B: the one from-scratch rerun the repair legitimately needs (checkpoints the
                                # buggy code wrote are untrustworthy) is not charged as lost work; every other attempt is


@dataclass
class FixOutcome:
    ok: bool                        # recovered correctly: fast path or invariants, lost work, ranks, no tampering
    bit_identical: bool             # every committed checkpoint bit-identical to the reference's
    params_match_reference: bool    # final committed parameters equal the reference's within TOLERANCE
    samples_match_reference: bool   # logical sample accounting (data server log + committed steps) equals the reference's
    restores_consistent: bool       # each resume's first committed state follows from the checkpoint it resumed from
    no_duplicate_updates: bool      # the final run reported each step in [resume, N) exactly once (evidence only)
    lost_steps: int                 # steps recomputed over every recovery attempt since the incident
    ranks_consistent: bool          # all ranks completed and ended with the same state digest
    tampered: tuple = ()            # non-editable files the agent changed (discarded)
    detail: str = ""
    steps_completed: int = 0
    ckpt_valid: bool = False        # the final checkpoint was committed
    sample_gaps: int = 0            # reference samples missing from the recovered history
    sample_dups: int = 0            # samples consumed more often than in the reference
    states_match_reference: bool = False   # fast path: full-state equality at every committed checkpoint
    transitions_consistent: bool = False   # every committed state follows from the previous one under the pristine update
    path: str = ""                  # state_equal | invariants | "" (not recovered)
    relaunches: int = 0             # recovery attempts since the incident (workspace and protocol)
    sandboxed: bool = True          # the re-execution ran under OS isolation; False -> invalid, never verified success
    exec_mode: str = "sandboxed"    # sandboxed | modal_sandbox | dev_container (unverified DEV training reward only)
    fault_not_injected: bool = False  # the incident's planned worker failure could not be carried out
    sampler_behaviour: bool = True  # v1.1 only: post-restore sampler positions follow the reference (always True under v1)
    history_intact: bool = True     # v1.2 only: no run re-committed an older checkpoint with new content (always True before)


def mark_sandbox(out: FixOutcome, sandboxed: bool) -> FixOutcome:
    """An unsandboxed re-execution is not a verification of agent-edited code: it can never count as success."""
    out.sandboxed = sandboxed
    if not sandboxed:
        logging.getLogger(__name__).warning("INVALID verification: re-execution ran without OS isolation; ok forced False")
        out.ok, out.path = False, ""
        out.detail = ("INVALID: unsandboxed re-execution; " + out.detail).strip("; ")
    return out


EXEC_SANDBOXED, EXEC_MODAL, EXEC_DEV_CONTAINER = "sandboxed", "modal_sandbox", "dev_container"
BACKENDS = (None, "modal")      # None: the local OS sandbox (jobs/sandbox.py); "modal": traindoctor_env/modal_backend.py


NO_RELAUNCH = FixOutcome(False, False, False, False, False, False, 0, False)


def lost_work(jd: Path, frontier: int, runs_before: set[str], scratch_allowance: int = 0) -> int:
    """Steps recomputed over EVERY relaunch after the incident, from the launcher's ledger: each run re-applies the
    steps in [resume, min(its end, furthest progress so far)). `scratch_allowance` runs from step 0 are exempt
    (Task B's one validation rerun); the progress they make then restarts the count for what follows."""
    lost = 0
    for rec in trusted.ledger(jd):
        if rec["run_id"] in runs_before:
            continue
        start = rec["start_step"]
        served = max((len(v) for v in rec["served"].values()), default=0)
        end = start + max(0, served - (0 if rec["completed"] else 1))   # the request in flight when it died was not a step
        if scratch_allowance and start == 0:
            scratch_allowance, frontier = scratch_allowance - 1, end
            continue
        lost += max(0, min(frontier, end) - start)
        frontier = max(frontier, end)
    return lost


# --- checkpoint states ----------------------------------------------------------------------------------------
def _ref_state(ref: dict, step: int) -> dict | None:
    st = ref["states"].get(str(step))
    return None if st is None else {k: torch.tensor(v) for k, v in st.items()}


def _load(jd: Path, cap: dict) -> dict | None:
    try:
        st = torch.load(trusted.trusted_dir(jd) / cap["file"], weights_only=True)
    except Exception:  # anything unreadable is simply not a valid state  # noqa: BLE001
        return None
    return st if isinstance(st, dict) and all(isinstance(v, torch.Tensor) for v in st.values()) else None


def _close(a: dict, b: dict, tol: float, keys=None) -> bool:
    """Equal states: the same keys (or, if given, at least those), same shapes, finite, counters exact, floats within tol."""
    if keys is None and set(a) != set(b):
        return False
    keys = set(b) if keys is None else set(keys)
    if not keys <= set(a):
        return False
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            return False
        if k in ("sampler", "step"):
            if not torch.equal(x, y):
                return False
        elif not (torch.isfinite(x).all() and (x.double() - y.double()).abs().max().item() <= tol if x.numel() else True):
            return False
    return True


def _batches(rec: dict, rank: int, a: int, b: int, rc: dict, salt: bytes):
    win = step_batches(rec["served"][str(rank)])
    lo, hi = a - rec["start_step"], b - rec["start_step"]
    if lo < 0 or any(k not in win for k in range(lo, hi)):
        return None
    spe = rc["num_samples"] // rc["world_size"] // rc["batch_size"]
    return [batch_tensors(salt_at(salt, win[k][0] * spe + win[k][1]), rc["seed"], win[k][2]) for k in range(lo, hi)]


def _replay(state: dict | None, a: int, b: int, rec: dict, rc: dict, salt: bytes) -> dict | None:
    """The pristine update rule (SGD, momentum 0.9, gradients averaged over ranks) applied to the batches the data
    server served for steps a..b-1, starting from the committed state at step a (or the initial model)."""
    per_rank = [_batches(rec, r, a, b, rc, salt) for r in range(rec["world"])]
    if any(p is None for p in per_rank):
        return None
    h = rc["hidden"]
    if state is None:
        torch.manual_seed(rc["seed"])
        fc1, fc2 = torch.nn.Linear(8, h), torch.nn.Linear(h, 4)
        params = [t.detach().clone() for t in (fc1.weight, fc1.bias, fc2.weight, fc2.bias)]
        bufs = [None] * 4
    else:
        names = ["model.fc1.weight", "model.fc1.bias", "model.fc2.weight", "model.fc2.bias"]
        if not all(n in state for n in names) or not all(f"opt.{i}" in state for i in range(4)):
            return None
        params, bufs = [state[n].clone() for n in names], [state[f"opt.{i}"].clone() for i in range(4)]
    for k in range(b - a):
        grads = [torch.zeros_like(p) for p in params]
        for r in range(rec["world"]):
            x, y = per_rank[r][k]
            ps = [p.clone().requires_grad_(True) for p in params]
            loss = torch.nn.functional.mse_loss(torch.relu(x @ ps[0].T + ps[1]) @ ps[2].T + ps[3], y)
            for g, gr in zip(grads, torch.autograd.grad(loss, ps)):
                g += gr
        for i in range(4):
            g = grads[i] / rec["world"]
            bufs[i] = g.clone() if bufs[i] is None else bufs[i] * 0.9 + g
            params[i] = params[i] - rc["lr"] * bufs[i]
    out = {k: v for k, v in zip(["model.fc1.weight", "model.fc1.bias", "model.fc2.weight", "model.fc2.bias"], params)}
    out.update({f"opt.{i}": bufs[i] for i in range(4)})
    return out


def _transitions(jd: Path, proto: list[dict], loaded: dict, rc: dict, salt: bytes) -> tuple[bool, bool]:
    """(all transitions follow the pristine update rule, those starting at a resume do). A transition goes from the
    committed state a run started from (or the initial model) to its first committed state, then between its
    consecutive committed states."""
    ledger = trusted.ledger(jd)
    all_ok = resume_ok = True
    for rec in proto:
        if rec["resume_from"] is None:
            prev_state, prev_step = None, 0
        else:
            # The baseline is what the launcher had captured BEFORE it launched this run (pinned by hash in the
            # ledger). Nothing the run itself, or any later run, wrote can stand in for it.
            idx = next(i for i, r in enumerate(ledger) if r["run_id"] == rec["run_id"])
            pin = rec.get("resume_sha256")
            src = next((c for r in reversed(ledger[:idx]) for c in reversed(r["captures"])
                        if c["ckpt_id"] == rec["resume_from"] and c["step"] == rec["start_step"] and c["sha256"] == pin), None)
            prev_state, prev_step = (_load(jd, src) if src else None), rec["start_step"]
            if src is None or prev_state is None:
                all_ok = resume_ok = False
                continue
        for c in sorted(rec["captures"], key=lambda c: c["step"]):
            st = loaded[(rec["run_id"], c["file"])]
            if c["step"] < rec["start_step"]:
                continue                       # an older checkpoint still listed in the manifest: not this run's work
            if c["step"] == rec["start_step"] and rec["resume_from"] is not None:
                all_ok &= st is not None and prev_state is not None and _close(st, prev_state, 0.0)   # the very checkpoint resumed from
                continue
            expected = _replay(prev_state, prev_step, c["step"], rec, rc, salt)
            ok = st is not None and expected is not None and _close(st, expected, TOLERANCE, expected.keys())
            all_ok &= ok
            if prev_step == rec["start_step"] and rec["resume_from"] is not None:
                resume_ok &= ok
            prev_state, prev_step = st, c["step"]
    return all_ok, resume_ok


def _format_ok(st: dict | None, c: dict, ref: dict, rc: dict) -> bool:
    """A committed checkpoint carries the model, optimizer momentum and a sampler position that is exactly where a
    job that took c.step steps stands; extra entries are allowed."""
    if st is None:
        return False
    spe = rc["num_samples"] // rc["world_size"] // rc["batch_size"]
    need = set(ref["states"]["1"])
    return (need <= set(st) and all(torch.isfinite(st[k]).all() for k in need if k not in ("sampler", "step"))
            and st["step"].tolist() == [c["step"]] and st["sampler"].tolist() == list(divmod(c["step"], spe)))


FORMAT_CHECKS = ("v1", "v1.1", "v1.2")   # v1 (default, frozen): checkpoints must STORE the sampler position. v1.1 (opt-in, NOT active): the position is judged by behaviour; see tests/test_verifier_v11.py. v1.2 (opt-in): v1.1 + history_intact


def _format_ok_v11(st: dict | None, c: dict, ref: dict, rc: dict) -> bool:
    """v1.1 representation check: model + optimizer momentum + step counter, finite, step == the captured step.
    A stored sampler position is NOT required; if one is present it must equal the reference position for this step
    (a conflicting or fabricated one is rejected). Extra entries are allowed."""
    if st is None:
        return False
    spe = rc["num_samples"] // rc["world_size"] // rc["batch_size"]
    need = set(ref["states"]["1"]) - {"sampler"}
    if not (need <= set(st) and all(torch.isfinite(st[k]).all() for k in need if k != "step")
            and st["step"].tolist() == [c["step"]]):
        return False
    return "sampler" not in st or st["sampler"].tolist() == list(divmod(c["step"], spe))


def sampler_behaviour_ok(proto: list[dict], ledger: list[dict], load, rc: dict, steps: int) -> bool:
    """v1.1 behavioural sampler check, from harness-owned records only (the data server's served log in the launcher's
    ledger and the checkpoint the launcher captured; never from anything the job wrote). For every run: the position
    each rank requested right after the restore equals the reference position for the committed step of the
    checkpoint resumed from (divmod(step, steps_per_epoch)), and from there it advances one reference step per
    acknowledged step with no repeat, skip or gap (a completed run covers every step up to the end, so training
    continued from the restore). The restored checkpoint must be the very one pinned in the ledger, carry step ==
    the run's start step, and every run must account for exactly the reference world's ranks."""
    spe = rc["num_samples"] // rc["world_size"] // rc["batch_size"]
    for rec in proto:
        if rec["world"] != rc["world_size"] or set(rec["served"]) != {str(r) for r in range(rc["world_size"])}:
            return False
        if rec["resume_from"] is None:
            base = 0
            if rec["start_step"] != 0:
                return False
        else:
            idx = next(i for i, r in enumerate(ledger) if r["run_id"] == rec["run_id"])
            pin = rec.get("resume_sha256")
            src = next((c for r in reversed(ledger[:idx]) for c in reversed(r["captures"])
                        if c["ckpt_id"] == rec["resume_from"] and c["step"] == rec["start_step"] and c["sha256"] == pin), None)
            st = load(src) if src else None
            if st is None or "step" not in st or st["step"].numel() != 1:
                return False
            base = int(st["step"])
            if base != rec["start_step"] or base != src["step"]:
                return False
        need = (steps - base) if rec["completed"] else 0
        for r in range(rc["world_size"]):
            win = step_batches(rec["served"][str(r)])
            if sorted(win) != list(range(len(win))) or len(win) < need:
                return False
            if any(tuple(win[k][:2]) != divmod(base + k, spe) for k in win):
                return False
    return True


def history_intact(proto: list[dict], ledger: list[dict]) -> bool:
    """v1.2: a checkpoint a run's manifest lists BELOW the step it started from is history, not that run's work, so
    _transitions skips it. It must then be byte-identical (same id, step and launcher hash) to a capture the launcher
    made in an EARLIER run; otherwise the run fabricated or rewrote an old checkpoint, which the state checks never see."""
    for rec in proto:
        idx = next(i for i, r in enumerate(ledger) if r["run_id"] == rec["run_id"])
        seen = {(c["ckpt_id"], c["step"], c["sha256"]) for r in ledger[:idx] for c in r["captures"]}
        if any((c["ckpt_id"], c["step"], c["sha256"]) not in seen for c in rec["captures"] if c["step"] < rec["start_step"]):
            return False
    return True


PARAM_NAMES = ("model.fc1.weight", "model.fc1.bias", "model.fc2.weight", "model.fc2.bias")


def state_digest_of(st: dict | None) -> str | None:
    """The digest jobs/runtime/harness.py reports per rank (parameters and momentum, in optimizer order), computed
    from a checkpoint the LAUNCHER captured. Rank-reported digests must equal this: identical forged digests do not."""
    if st is None or not all(n in st and f"opt.{i}" in st for i, n in enumerate(PARAM_NAMES)):
        return None
    h = hashlib.sha256()
    for i, n in enumerate(PARAM_NAMES):
        h.update(st[n].detach().contiguous().numpy().tobytes())
        h.update(st[f"opt.{i}"].detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def world_joined(digests: dict, rc: dict) -> bool:
    """I6: the final run's launcher-recorded ranks (one digest slot per rank the LAUNCHER started) equal the reference
    world size. A repair that shrinks world_size (drop a rank, collapse to one process) can never pass, even if its
    numerics happen to match."""
    return len(digests) == rc["world_size"]


def check(jd: Path, snap: Snapshot, lost: int, tampered: tuple = (), before: frozenset = frozenset(),
          expect_kill: bool = False, relaunches: int = 0, ref: dict | None = None, salt=None,
          format_check: str = "v1") -> FixOutcome:
    """`ref` and `salt` are the recovery target and data keying of THIS verification run (a fresh SaltSchedule and
    the reference computed under it); they default to the snapshot's own, which only exploratory runs use."""
    if format_check not in FORMAT_CHECKS:
        raise ValueError(f"unknown format_check {format_check!r}")
    ref = snap.reference if ref is None else ref
    salt = snap.salt if salt is None else salt
    rc = ref["config"]
    steps, ckpt_every = ref["steps"], rc["ckpt_every"]
    led = trusted.ledger(jd)
    proto = [r for r in led if r["run_id"] not in before]
    base = dict(lost_steps=lost, tampered=tuple(tampered), relaunches=relaunches)
    if not proto or not proto[-1]["completed"]:
        return dataclasses.replace(NO_RELAUNCH, **base)
    if expect_kill and not proto[0]["kill_done"]:
        if proto[0].get("fault_not_injected"):
            return dataclasses.replace(NO_RELAUNCH, detail="fault plan not carried out", fault_not_injected=True, **base)
        return dataclasses.replace(NO_RELAUNCH, detail="the incident's worker failure did not happen", **base)
    last = proto[-1]
    caps = [(r, c) for r in proto for c in r["captures"]]
    loaded = {(r["run_id"], c["file"]): _load(jd, c) for r, c in caps}
    final = next((c for c in last["captures"] if c["step"] == steps), None)
    refs = {c["file"]: _ref_state(ref, c["step"]) for _, c in caps}
    fast = bool(caps) and final is not None and all(
        loaded[(r["run_id"], c["file"])] is not None and refs[c["file"]] is not None
        and _close(loaded[(r["run_id"], c["file"])], refs[c["file"]], FAST_TOL) for r, c in caps)
    exact = bool(caps) and all(
        loaded[(r["run_id"], c["file"])] is not None and refs[c["file"]] is not None
        and _close(loaded[(r["run_id"], c["file"])], refs[c["file"]], 0.0) for r, c in caps)
    fin_st, fin_ref = (loaded[(last["run_id"], final["file"])] if final else None), _ref_state(ref, steps)
    params_match = fin_st is not None and _close(fin_st, fin_ref, TOLERANCE, [k for k in fin_ref if k.startswith("model.")])
    history = format_check != "v1.2" or history_intact(proto, led)
    if format_check == "v1":
        fmt, behaviour = bool(caps) and all(_format_ok(loaded[(r["run_id"], c["file"])], c, ref, rc) for r, c in caps), True
    else:
        fmt = bool(caps) and all(_format_ok_v11(loaded[(r["run_id"], c["file"])], c, ref, rc) for r, c in caps)
        behaviour = bool(caps) and sampler_behaviour_ok(proto, led, lambda cap: _load(jd, cap), rc, steps)
    trans, resumes = _transitions(jd, proto, loaded, rc, salt) if caps else (False, False)
    lin = lineage(jd, steps)
    want = {int(r): {int(s): ids for s, ids in d.items()} for r, d in ref["samples"].items()}
    w = Counter(i for d in want.values() for ids in d.values() for i in ids)
    g = Counter(i for d in lin.values() for ids in d.values() for i in ids)
    samples_ok = want.keys() == lin.keys() and all({s: sorted(v) for s, v in want[r].items()} == {s: sorted(v) for s, v in lin[r].items()} for r in want)
    ds = [d for d in last["digests"].values()]
    trusted_digest = state_digest_of(fin_st)    # from the checkpoint the launcher captured, not from anything a rank reports
    consistent = trusted_digest is not None and bool(ds) and all(d == trusted_digest for d in ds)
    applied_once = all(a == steps - last["start_step"] for a in last["acked"].values())
    ends_at_steps = max((c["step"] for c in last["captures"]), default=0) == steps   # no run past the reference's end
    joined = world_joined(last["digests"], rc)
    ok_base = final is not None and ends_at_steps and lost <= ckpt_every and consistent and not tampered and joined
    # Sample coverage and every state transition must hold on BOTH paths: equality with the reference alone (the fast
    # path) never stands in for having trained on the data this run was served.
    ok = ok_base and samples_ok and trans and behaviour and history and (fast or fmt)
    return FixOutcome(ok, exact, params_match, samples_ok, resumes, applied_once, lost, consistent, tuple(tampered),
                      "" if joined else "I6 world size differs from the reference",
                      steps, final is not None, sum((w - g).values()), sum((g - w).values()), fast, trans,
                      ("state_equal" if fast else "invariants") if ok else "", relaunches, sampler_behaviour=behaviour,
                      history_intact=history)


def _fresh_protocol_dir(snap: Snapshot, ws_root: Path, tmp: Path) -> tuple[Path, set[str], int, dict, tuple]:
    """A fresh copy of the incident with ONLY the agent's changes to EDITABLE files applied (changes to any other
    file under job/ are discarded and reported). Private material is never copied. Returns (job_dir, runs that
    predate the protocol, frontier at its start, resubmit kwargs, tampered paths)."""
    meta, pristine = snap.meta, snap.fixture_dir / "job"
    jd = tmp / "protocol"
    try:
        ws_names = set(safefs.listdir(ws_root, "job"))
    except OSError:
        ws_names = set()    # job/ replaced by a symlink or removed: every non-editable pristine file then counts as changed
    names = ws_names | {p.name for p in pristine.glob("*")}
    tampered = [f"job/{n}" for n in names if f"job/{n}" not in EDITABLE and not _same(ws_root, f"job/{n}", pristine / n)]
    if meta["task"] == "A":
        trusted.copy_job(snap.fixture_dir, jd)
        base, before, frontier, kw = jd / "job", frozenset(run_names(jd)), meta["frontier"], {}
    else:
        jd.mkdir()
        shutil.copytree(pristine, jd / "job")
        shutil.copytree(snap.fixture_dir / "data", jd / "data")
        base, before, frontier, kw = jd / "job", frozenset(), 0, dict(kill=tuple(meta["kill"]), restart_on_failure=True)
    for rel in EDITABLE:
        try:
            blob = safefs.read_bytes(ws_root, rel)
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            tampered.append(rel)    # a symlink or special file where an editable file should be: discarded, reported
            continue
        (base / Path(rel).name).write_bytes(blob)
    return jd, before, frontier, kw, tuple(sorted(tampered))


def _same(ws_root: Path, rel: str, pristine: Path) -> bool:
    """The workspace file equals the pristine one (both absent counts). The workspace side is read without
    following symlinks; anything but a plain regular file there differs."""
    try:
        mine = safefs.read_bytes(ws_root, rel)
    except FileNotFoundError:
        mine = None
    except (OSError, ValueError):
        return False
    return mine == (pristine.read_bytes() if pristine.exists() else None)


class Replay:
    """The agent's workspace. Edits are plain file writes under job/; relaunch goes through resubmit."""

    def __init__(self, snapshot: Snapshot, allow_unsandboxed: bool | None = None, backend: str | None = None,
                 format_check: str = "v1"):
        if backend not in BACKENDS:
            raise ValueError(f"unknown replay backend {backend!r}")
        if format_check not in FORMAT_CHECKS:
            raise ValueError(f"unknown format_check {format_check!r}")
        self.format_check = format_check
        self.backend = backend
        self.backend_info: dict = {}
        self.snapshot = snapshot
        self.allow_unsandboxed = ALLOW_UNSANDBOXED if allow_unsandboxed is None else allow_unsandboxed
        self._tmp = tempfile.TemporaryDirectory()
        self.jd = Path(self._tmp.name) / "workspace"
        trusted.copy_job(snapshot.fixture_dir, self.jd)
        self.runs0 = frozenset(run_names(self.jd))
        self.relaunched = False
        self.notes: list[str] = []

    def close(self) -> None:
        self._tmp.cleanup()

    def write_file(self, rel: str, text: str) -> bool:
        parts = rel.split("/")
        if len(parts) != 2 or parts[0] != "job":
            self.notes.append("write_file rejected (outside job/)")
            return False
        try:
            safefs.write_bytes(self.jd, rel, text.encode())
        except (OSError, ValueError):
            self.notes.append("write_file rejected (unsafe path)")
            return False
        return True

    def edit(self, e: Edit) -> None:
        apply_edits(self.jd, [e])

    def relaunch(self, resume_from: str | None = None, chaos: bool = False) -> None:
        kill = tuple(self.snapshot.meta["kill"]) if chaos and self.snapshot.meta["kill"] else None
        resubmit(self.jd, resume_from, kill=kill, restart_on_failure=bool(kill), sandbox=True, salt=self.snapshot.salt,
                 allow_unsandboxed=self.allow_unsandboxed)
        self.relaunched = True

    def outcome(self) -> FixOutcome:
        return self.outcomes((self.format_check,))[self.format_check]

    def outcomes(self, format_checks: tuple) -> dict[str, FixOutcome]:
        """One re-execution of the final job/, judged under each of `format_checks` (the run is shared, so the verdicts
        differ only in the checks). outcome() is outcomes((self.format_check,))."""
        if not format_checks or any(fc not in FORMAT_CHECKS for fc in format_checks):
            raise ValueError(f"unknown format_check in {format_checks!r}")
        snap = self.snapshot
        task_b = snap.meta["task"] == "B"
        with tempfile.TemporaryDirectory() as tmp:
            jd, before, frontier, kw, tampered = _fresh_protocol_dir(snap, self.jd, Path(tmp))
            # A per-verification data challenge: from the first position this run will compute, the sample VALUES are
            # keyed by a nonce nobody (job code, earlier runs) has seen, so nothing produced earlier can be valid here.
            start = manifest.latest_valid(jd, manifest.load(jd)) if not task_b else None
            sched = SaltSchedule.fresh_for(snap.salt, start["step"] if start else 0)
            ref = reference_for_config(snap.reference["pristine_config"], sched)
            if self.backend == "modal":
                from traindoctor_env import modal_backend as sandbox_backend
                self.backend_info = sandbox_backend.run_resubmit(jd, sched, kw)
            else:
                resubmit(jd, sandbox=True, salt=sched, allow_unsandboxed=self.allow_unsandboxed, **kw)
            lost = max(lost_work(jd, frontier, before),
                       lost_work(self.jd, snap.meta["frontier"], self.runs0, B_VALIDATION_SCRATCH_RUNS if task_b else 0))
            n = len(run_names(jd)) - len(before) + len(run_names(self.jd)) - len(self.runs0)
            outs = {fc: check(jd, snap, lost, tampered, before, expect_kill=task_b, relaunches=n, ref=ref, salt=sched,
                              format_check=fc) for fc in format_checks}
        return {fc: self._label(out) for fc, out in outs.items()}

    def _label(self, out: FixOutcome) -> FixOutcome:
        out.detail = out.detail or "; ".join(self.notes)
        if self.backend == "modal":
            # Only this path sets the label, after run_resubmit returned: the job ran in a network-less gVisor sandbox.
            out.sandboxed, out.exec_mode = True, EXEC_MODAL
            return out
        if DEV_CONTAINER_MODE and self.allow_unsandboxed and not sandbox.available():
            out.sandboxed, out.exec_mode = False, "dev_container"   # real outcome, kept unverified: strict V2 still rejects it
            return out
        return mark_sandbox(out, sandbox.available())


def verify_fix_checks(snapshot: Snapshot, edits: list[Edit], format_checks: tuple, allow_unsandboxed: bool | None = None,
                      backend: str | None = None) -> dict[str, FixOutcome]:
    """verify_fix judged under several format checks from ONE re-execution."""
    rp = Replay(snapshot, allow_unsandboxed, backend)
    try:
        for e in edits:
            rp.edit(e)
        return rp.outcomes(tuple(format_checks))
    finally:
        rp.close()


def verify_fix(snapshot: Snapshot, edits: list[Edit], allow_unsandboxed: bool | None = None,
               backend: str | None = None, format_check: str = "v1") -> FixOutcome:
    rp = Replay(snapshot, allow_unsandboxed, backend, format_check)
    try:
        for e in edits:
            rp.edit(e)
        return rp.outcome()
    finally:
        rp.close()
