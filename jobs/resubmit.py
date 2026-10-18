"""Resubmission script: owns job_dir state (.lock, ckpt/MANIFEST.json, runs/<run_id>/, scheduler.jsonl).
Every relaunch (the agent's `relaunch`, the scheduler's automatic restart, verification) goes through resubmit()."""
from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path

from jobs import manifest, safefs, trusted
from jobs.launcher import BadConfig, JobResult, launch_params, run_job

SCRATCH = "none"  # resume_from value meaning "start from step 0"


class LockHeld(Exception):
    pass


@dataclass
class ResubmitResult:
    ok: bool
    status: str  # completed | failed | watchdog_killed | refused
    run_id: str | None
    error_class: str | None
    message: str
    job: JobResult | None = None


def event(job_dir: Path, kind: str, detail: str) -> None:
    trusted.append_event(Path(job_dir), kind, detail)


def _runs(job_dir: Path) -> list[str]:
    try:
        return safefs.listdir(job_dir, "runs")
    except FileNotFoundError:
        return []


def lock_is_live(lock: Path) -> bool:
    """A lock is live only while its owner holds an flock on it. Liveness is a property of the open file, not
    of the pid written inside, so a copied/frozen lock is stale even if its pid now belongs to another process."""
    try:
        fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False   # a symlink (or anything unopenable) is nobody's live lock
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _acquire_lock(job_dir: Path, run_id: str) -> int:
    """Create .lock (existence = held, as a scheduler sees it) and flock it; returns the owning fd."""
    lock = job_dir / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError:
        try:
            held = json.loads(safefs.read_bytes(job_dir, ".lock", 4096) or b"{}")
        except (OSError, ValueError):
            held = {}
        raise LockHeld(f"lock held by run_id={held.get('run_id')} pid={held.get('pid')}") from None
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.write(fd, json.dumps(dict(pid=os.getpid(), run_id=run_id)).encode())
    return fd


def clean_stale_state(job_dir: Path) -> bool:
    """Remove the lock if nobody holds it; never touches checkpoints. Returns whether a lock was removed."""
    lock = Path(job_dir) / ".lock"
    if os.path.lexists(lock) and not lock_is_live(lock):
        lock.unlink()
        event(Path(job_dir), "clean_stale_state", "stale lock removed")
        return True
    return False


def resubmit(job_dir: Path, resume_from: str | None = None, kill: tuple[int, int] | None = None,
             restart_on_failure: bool = False, sandbox: bool = False, salt: bytes | None = None,
             allow_unsandboxed: bool = False) -> ResubmitResult:
    """Relaunch the job. resume_from=None resumes from the latest valid checkpoint (SCRATCH = step 0).
    kill=(rank, step) fails one worker at that step boundary; with restart_on_failure the scheduler then restarts the
    job once from the latest valid checkpoint (the second result is returned). sandbox=True runs the job's
    processes under the OS sandbox (jobs/sandbox.py); salt keys the data server (jobs/dataserver.py)."""
    job_dir = Path(job_dir)
    try:
        launch_params(job_dir)
    except BadConfig as e:
        event(job_dir, "resubmit_refused", str(e))
        return ResubmitResult(False, "refused", None, "BAD_CONFIG", str(e))
    entries = manifest.load(job_dir)
    if resume_from is None:
        e = manifest.latest_valid(job_dir, entries)
        resume_from = e["ckpt_id"] if e and isinstance(e.get("ckpt_id"), str) and isinstance(e.get("step"), int) else SCRATCH
    run_id = f"run{max(len(_runs(job_dir)), len(trusted.run_ids(job_dir)))}"
    try:
        lock_fd = _acquire_lock(job_dir, run_id)
    except LockHeld as e:
        event(job_dir, "resubmit_refused", str(e))
        return ResubmitResult(False, "refused", None, "LOCK_HELD", str(e))
    start_step = next((e["step"] for e in entries if e.get("ckpt_id") == resume_from and isinstance(e.get("step"), int)), 0)
    safefs.makedirs(job_dir, f"runs/{run_id}", exist_ok=False)
    safefs.write_bytes(job_dir, f"runs/{run_id}/start.json", json.dumps(
        dict(run_id=run_id, resume_from=None if resume_from == SCRATCH else resume_from, started_at_step=start_step),
        sort_keys=True).encode())
    event(job_dir, "submitted", f"{run_id} resume_from={resume_from}")
    try:
        job = run_job(job_dir, run_id, None if resume_from == SCRATCH else resume_from, start_step, kill,
                      sandbox_jobs=sandbox, salt=salt, allow_unsandboxed=allow_unsandboxed)
        (job_dir / ".lock").unlink(missing_ok=True)
        event(job_dir, job.status, run_id + (f" {job.error_class}" if job.error_class else ""))
    finally:
        os.close(lock_fd)
    res = ResubmitResult(job.status == "completed", job.status, run_id, job.error_class, job.status, job)
    if kill and restart_on_failure and job.status == "failed":
        event(job_dir, "auto_restart", f"{run_id} lost a worker; restarting from the latest checkpoint")
        return resubmit(job_dir, sandbox=sandbox, salt=salt, allow_unsandboxed=allow_unsandboxed)
    return res
