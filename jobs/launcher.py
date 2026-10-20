"""Run job/train.py once per rank (plain subprocesses, cwd = the job directory, started through
jobs/runtime/harness.py), serve their data, poll them ourselves, enforce a no-progress watchdog, optionally fail
one worker at a step boundary, and return per-rank results. Never hangs.

Everything a verdict needs is recorded HERE, in the launcher's process, in the ledger (jobs/trusted.py): the
launcher's own process history, the data server's log, and a byte copy of every checkpoint taken when it was
committed. Nothing is read back from files the job writes. Exceptions are classified from each rank's stderr.

The step-boundary rendezvous: to fail a worker (kill=(rank, step)) deterministically, every rank is parked when
it asks for the batch after `step` (so every checkpoint written so far is complete and nothing is running), the
launcher captures checkpoints, kills the victim, waits for it to be dead, and only then releases its peers."""
from __future__ import annotations

import json
import logging
import os
import re
import selectors
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from jobs import safefs, sandbox, trusted
from jobs.config import BadConfig, MAX_WORLD, job_config, validate_config
from jobs.dataserver import DataServer, DataSpec, salt_for

log = logging.getLogger(__name__)
RUNTIME_HARNESS = sandbox.RUNTIME / "harness.py"
TIMEOUT_MARKERS = ("Timed out", "Connection closed by peer", "Connection reset", "Broken pipe")
STARTUP_GRACE_S = 30.0
MAX_RUNTIME_S = 120.0
VICTIM_DEAD_S = 10.0
# Environment handed to job processes: an explicit allowlist, never the launcher's own environment.
ENV_PASSTHROUGH = ("PATH",)


@dataclass
class RankResult:
    rank: int
    exitcode: int | None
    status: str  # ok | error | killed
    exc_class: str | None
    exc_msg: str | None
    last_step: int
    digest: str | None = None  # state digest the rank reported after its last step (evidence), completed ranks only


@dataclass
class JobResult:
    run_id: str
    status: str  # completed | failed | watchdog_killed | refused
    ranks: list[RankResult]
    last_step: int
    error_class: str | None
    fault_not_injected: bool = False   # a planned worker failure named a rank that does not exist in this config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def loopback_interface() -> str:
    """The loopback interface's name on this machine (lo0 on macOS, lo on Linux): derived, not assumed."""
    names = [n for _, n in socket.if_nameindex()]
    return next(n for n in ("lo0", "lo") if n in names)


def classify(stderr: str, pg_timeout_s: float) -> tuple[str, str]:
    """Peers of a hung collective see either a timeout or a closed connection; both are the same event."""
    if any(m in stderr for m in TIMEOUT_MARKERS):
        return "COLLECTIVE_TIMEOUT", f"collective timed out (timeout={pg_timeout_s:g}s)"
    last = next((x for x in reversed(stderr.splitlines()) if x.strip()), "")
    return "PYTHON_ERROR", re.sub(r"\s+", " ", last)[:200]


def _progress(job_dir: Path, run_id: str) -> int:
    """Bytes of training progress written so far (collective traces); the data server's activity is added by the caller."""
    try:
        names = [n for n in safefs.listdir(job_dir, f"runs/{run_id}/trace") if re.fullmatch(r"rank\d+\.jsonl", n)]
    except OSError:
        return 0
    return sum(safefs.file_size(job_dir, f"runs/{run_id}/trace/{n}") for n in names)


def job_env(rank: int, world: int, port: int, run_id: str, resume_from: str | None, start_step: int, fd: int) -> dict:
    env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank), RUN_ID=run_id,
               RESUME_FROM=resume_from or "", HARNESS_START=str(start_step), TD_FD=str(fd),
               GLOO_SOCKET_IFNAME=loopback_interface(), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0",
               PYTHONDONTWRITEBYTECODE="1")
    return env


MAX_OUT_BYTES = 4 << 20     # replies queued for one rank; a job that lets more pile up is not reading
SEND_DEADLINE_S = 5.0       # queued replies must make progress within this long


class _Rank:
    """One job process's data socket. Output is queued and written nonblocking, so a job that does not read its
    replies can never block the launcher (and with it the watchdog): it overflows or goes stuck, and is killed."""

    def __init__(self, sock: socket.socket):
        self.sock, self.buf, self.parked, self.eof = sock, b"", None, False
        self.out = bytearray()
        self.overflow = False
        self.progress_at = time.monotonic()

    def send(self, msg: dict) -> None:
        data = json.dumps(msg).encode() + b"\n"
        if len(self.out) + len(data) > MAX_OUT_BYTES:
            self.overflow = True
            return
        if not self.out:
            self.progress_at = time.monotonic()
        self.out += data
        self.flush()

    def flush(self) -> None:
        while self.out:
            try:
                n = self.sock.send(self.out)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                self.eof, self.out = True, bytearray()
                return
            del self.out[:n]
            self.progress_at = time.monotonic()

    def stuck(self, now: float) -> bool:
        return bool(self.out) and now - self.progress_at > SEND_DEADLINE_S


def launch_params(job_dir: Path) -> tuple[int, float, float, DataSpec]:
    try:
        cfg = job_config(job_dir)
        validate_config(cfg)
        world, pg, wd = cfg["world_size"], cfg["pg_timeout_s"], cfg["watchdog_s"]
        spec = DataSpec(world, cfg["data"]["num_samples"], cfg["batch_size"], cfg["seed"])
    except BadConfig:
        raise
    except Exception as e:  # any parse/shape problem is the same refusal  # noqa: BLE001
        raise BadConfig(f"unreadable job config: {type(e).__name__}") from None
    return world, pg, wd, spec


def run_job(job_dir: Path, run_id: str = "run0", resume_from: str | None = None, start_step: int = 0,
            kill: tuple[int, int] | None = None, on_spawn: Callable[[int, int], None] | None = None,
            sandbox_jobs: bool = False, salt: bytes | None = None,
            allow_unsandboxed: bool = False) -> JobResult:
    """kill=(rank, step): fail that rank once it has committed `step` (a worker failure), at the rendezvous."""
    job_dir = Path(job_dir)
    try:
        world, pg_timeout, watchdog, spec = launch_params(job_dir)
    except BadConfig as e:
        trusted.append_event(job_dir, "launcher_refused", str(e))
        return JobResult(run_id, "refused", [], -1, "BAD_CONFIG")
    safefs.makedirs(job_dir, f"runs/{run_id}/stderr")
    port = free_port()
    server = DataServer(spec, salt or salt_for("default"))
    cap = trusted.Capturer(job_dir, run_id)
    # The checkpoint this run resumes from, pinned by content hash before anything of it is launched.
    resume_sha = next((c["sha256"] for r in reversed(trusted.ledger(job_dir)) for c in reversed(r["captures"])
                       if resume_from and c["ckpt_id"] == resume_from and c["step"] == start_step), None)
    argv = [sys.executable, str(RUNTIME_HARNESS)]
    if sandbox_jobs:
        argv = sandbox.wrap(argv, job_dir, allow_unsandboxed)
    errs = [safefs.open_write(job_dir, f"runs/{run_id}/stderr/rank{r}.txt") for r in range(world)]
    ranks: list[_Rank] = []
    procs: list[subprocess.Popen] = []
    sel = selectors.DefaultSelector()
    for r in range(world):
        parent, child = socket.socketpair()
        procs.append(subprocess.Popen(argv, cwd=job_dir, pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=errs[r],
                                      env=job_env(r, world, port, run_id, resume_from, start_step, child.fileno())))
        child.close()
        parent.setblocking(False)
        ranks.append(_Rank(parent))
        sel.register(parent, selectors.EVENT_READ, r)
    if on_spawn:
        on_spawn(procs[0].pid, port)

    # A kill plan naming a rank this config does not have (e.g. a candidate edit that shrinks world_size below the
    # incident's victim rank) cannot be carried out. It is NEVER silently skipped: the run is flagged
    # fault_not_injected and verification then reports ok=False ("fault plan not carried out"). The menu's
    # world-shrink ("exclude_rank") is a destructive decoy, not a legitimate repair, so there is no alternative
    # semantics to preserve; the job simply runs without the failure so the flag is the only effect.
    fault_not_injected = bool(kill) and not 0 <= kill[0] < world
    if fault_not_injected:
        kill = None
    target = (kill[1] + 1 - start_step) if kill else None   # batches the victim is served before the failure
    state = dict(killed=False, rendezvous=bool(kill) and target is not None and target > 0)
    killed_by_us: set[int] = set()
    dropped: set[int] = set()   # ranks cut off for abusing the data protocol
    t0 = time.monotonic()
    last_change, seen = t0, 0

    def fail_victim_and_release() -> None:
        cap.scan()
        victim = kill[0]   # validated above: 0 <= victim < world
        procs[victim].kill()
        try:
            procs[victim].wait(VICTIM_DEAD_S)
        except subprocess.TimeoutExpired:
            pass
        killed_by_us.add(victim)
        state.update(killed=True, rendezvous=False)
        release()

    def release() -> None:
        state["rendezvous"] = False
        for r, rk in enumerate(ranks):
            if rk.parked is not None:
                req, rk.parked = rk.parked, None
                if procs[r].poll() is None:
                    rk.send(server.serve(r, req))
                process_lines(r)

    def on_line(r: int, line: bytes) -> None:
        rk = ranks[r]
        kind, msg = server.parse(r, line)
        if kind == "drop":
            rk.eof = True
            try:
                sel.unregister(rk.sock)
            except (KeyError, ValueError):
                pass
            procs[r].kill()
            dropped.add(r)
        elif kind == "error":
            rk.send(msg)
        elif kind == "step":
            server.note_step(r, msg)
        else:  # batch
            if r == 0:
                cap.scan()   # rank 0 wrote the checkpoint before the barrier that precedes this request
            if state["rendezvous"] and len(server.served[r]) == target:
                rk.parked = msg
                if all(x.parked is not None for x in ranks):
                    fail_victim_and_release()
            else:
                rk.send(server.serve(r, msg))

    def process_lines(r: int) -> None:
        rk = ranks[r]
        while rk.parked is None and b"\n" in rk.buf and not rk.eof:
            line, rk.buf = rk.buf.split(b"\n", 1)
            on_line(r, line)
        if rk.parked is None and len(rk.buf) > 4096 and not rk.eof:  # an unterminated oversized request
            on_line(r, rk.buf)
            rk.buf = b""

    try:
        while any(p.poll() is None for p in procs):
            now = time.monotonic()
            events = sel.select(0.002) if sel.get_map() else (time.sleep(0.002) or [])
            for key, _ in events:
                r = key.data
                try:
                    data = ranks[r].sock.recv(65536)
                except BlockingIOError:
                    continue
                except OSError:
                    data = b""
                if not data:
                    ranks[r].eof = True
                    sel.unregister(ranks[r].sock)
                else:
                    ranks[r].buf += data
                    process_lines(r)
            if state["rendezvous"] and any(p.poll() is not None for p in procs):
                release()   # a rank is gone before the boundary: the failure plan cannot be carried out
            for r, rk in enumerate(ranks):
                rk.flush()
                if (rk.overflow or rk.stuck(now)) and r not in dropped and procs[r].poll() is None:
                    rk.out, rk.eof = bytearray(), True    # not reading its replies: cut it off, the episode fails
                    try:
                        sel.unregister(rk.sock)
                    except (KeyError, ValueError):
                        pass
                    procs[r].kill()
                    dropped.add(r)
            p_now = _progress(job_dir, run_id) + server.activity
            if p_now != seen:
                seen, last_change = p_now, now
            if now - (last_change if seen > 0 else t0 + STARTUP_GRACE_S) > watchdog or now - t0 > MAX_RUNTIME_S:
                for r, q in enumerate(procs):
                    if q.poll() is None:
                        q.kill()
                        killed_by_us.add(r)
                break
    finally:
        for q in procs:
            if q.poll() is None:
                q.kill()
            q.wait(10)
        for f in errs:
            f.close()
        deadline = time.monotonic() + 1.0   # collect acks still in flight; a grandchild holding the socket cannot stall us
        for r, rk in enumerate(ranks):
            rk.sock.setblocking(False)
            while not rk.eof and time.monotonic() < deadline:
                try:
                    data = rk.sock.recv(65536)
                except OSError:
                    break
                if not data:
                    break
                rk.buf += data
            for line in rk.buf.split(b"\n"):
                kind, msg = server.parse(r, line) if line else ("drop", {})
                if kind == "step":
                    server.note_step(r, msg)
            rk.sock.close()
        server.close()
        sel.close()
        cap.scan()   # every process of this run is dead: nothing can change a checkpoint under us now

    results = []
    for r in range(world):
        stderr = trusted.read_nofollow(job_dir, f"runs/{run_id}/stderr/rank{r}.txt").decode(errors="replace")
        acked = server.acked[r]
        if r in dropped:
            status, cls, msg = "error", "PROTOCOL_VIOLATION", "cut off for abusing the data service (oversized request or unread replies)"
        elif r in killed_by_us:
            status, cls, msg = "killed", None, None
        elif procs[r].returncode == 0:
            status, cls, msg = "ok", None, None
        else:
            status = "error"
            cls, msg = classify(stderr, pg_timeout) if stderr.strip() else ("PYTHON_ERROR", f"exited with code {procs[r].returncode}")
        results.append(RankResult(r, procs[r].returncode, status, cls, msg, start_step + acked - 1 if acked else -1,
                                  server.digests[r] if status == "ok" else None))
        first_in_window = {}   # the agent's readable copy of what was served: the first batch of each step window
        for epoch, index, ids, win in server.served[r]:
            first_in_window.setdefault(win, ids)
        trusted.write_run_record(job_dir, run_id, f"samples/rank{r}.jsonl", "".join(
            json.dumps(dict(step=start_step + w, ids=first_in_window[w])) + "\n" for w in range(acked) if w in first_in_window))
    if any(x.status == "error" for x in results):
        status = "failed"
    elif any(x.status == "killed" for x in results):
        status = "watchdog_killed" if not kill else "failed"
    else:
        status = "completed"
    result = JobResult(run_id, status, results, max(x.last_step for x in results),
                       next((x.exc_class for x in results if x.exc_class), None), fault_not_injected)
    trusted.write_run_record(job_dir, run_id, "job.json", json.dumps(asdict(result), indent=1, sort_keys=True))
    trusted.append_run(job_dir, dict(
        run_id=run_id, start_step=start_step, resume_from=resume_from, resume_sha256=resume_sha, status=status, completed=status == "completed",
        world=world, kill_planned=bool(kill) or fault_not_injected, kill_done=state["killed"], fault_not_injected=fault_not_injected, sandboxed=sandbox_jobs,
        served={str(r): [[e, i, ids, win] for e, i, ids, win in server.served[r]] for r in range(world)},
        acked={str(r): server.acked[r] for r in range(world)},
        digests={str(r): server.digests[r] for r in range(world)}, captures=cap.records))
    return result
