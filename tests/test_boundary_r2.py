"""Boundary review r2 (an internal boundary review). No job is launched here.

Rules: (1) trusted code reads job-writable files only through jobs/safefs; (2) a verification run's resume baseline
is the checkpoint the launcher captured BEFORE launching it, never anything the run itself wrote; (3) the launcher's
I/O toward a job never blocks, so a job that does not read cannot freeze the launcher or its watchdog."""
import time
from types import SimpleNamespace

import torch

from jobs import launcher, trusted
from jobs.trusted import trusted_dir
from traindoctor_env import reexec
from traindoctor_env.snapshot import job_files
from traindoctor_env.tools import Episode

CONFIG = "world_size: 2\npg_timeout_s: 3\nwatchdog_s: 8\n"


def _linked_job(tmp_path):
    secret = tmp_path / "secret_key"
    secret.write_text("SECRETVALUE")
    jd = tmp_path / "jd"
    (jd / "job").mkdir(parents=True)
    (jd / "job" / "config.yaml").symlink_to(secret)
    (jd / "job" / "train.py").write_text("print('ok')\n")
    return jd


def test_job_config_tool_never_follows_a_symlinked_config(tmp_path):
    jd = _linked_job(tmp_path)
    ep = Episode.__new__(Episode)
    ep.replay, ep.snap = None, SimpleNamespace(fixture_dir=jd)
    assert "SECRETVALUE" not in ep._job_config()


def test_captured_job_files_never_follow_a_symlink(tmp_path):
    files = job_files(_linked_job(tmp_path))
    assert "job/train.py" in files and "job/config.yaml" not in files
    assert "SECRETVALUE" not in str(files)


# --- resume forgery ------------------------------------------------------------------------------------------
def _state(v: float, step: int) -> dict:
    st = {f"opt.{i}": torch.full((2,), v) for i in range(4)}
    st.update({n: torch.full((2,), v) for n in reexec.PARAM_NAMES})
    st.update(sampler=torch.tensor([0, step]), step=torch.tensor([step]))
    return st


def _cap(jd, run, name, st, ckpt_id="ck1", step=1, sha="aa"):
    f = trusted_dir(jd) / "captures" / run / name
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save(st, f)
    return dict(ckpt_id=ckpt_id, step=step, sha256=sha, file=str(f.relative_to(trusted_dir(jd))))


def _resume_case(tmp_path, pin="aa"):
    jd = tmp_path / "jd"
    jd.mkdir()
    genuine, forged = _state(1.0, 1), _state(0.0, 1)    # the job overwrote the resumed checkpoint with zeros
    c0 = _cap(jd, "run0", "0.pt", genuine, sha="aa")
    c1 = _cap(jd, "run1", "0.pt", forged, sha="bb")
    trusted.append_run(jd, dict(run_id="run0", start_step=0, resume_from=None, captures=[c0]))
    rec = dict(run_id="run1", start_step=1, resume_from="ck1", resume_sha256=pin, captures=[c1])
    trusted.append_run(jd, rec)
    loaded = {("run1", c1["file"]): forged}
    return jd, [rec], loaded


def test_a_resumed_checkpoint_the_run_overwrote_is_not_its_own_baseline(tmp_path):
    """Rule: recovery is judged from the state the launcher captured before the run started. A run that rewrites the
    checkpoint it resumed from (zeroed model and momentum) must not be accepted by comparing it with itself."""
    jd, proto, loaded = _resume_case(tmp_path)
    all_ok, resume_ok = reexec._transitions(jd, proto, loaded, {}, b"")
    assert not all_ok


def test_a_resume_with_no_matching_prelaunch_capture_is_refused(tmp_path):
    jd, proto, loaded = _resume_case(tmp_path, pin="nope")
    assert reexec._transitions(jd, proto, loaded, {}, b"")[0] is False


def test_an_honest_resume_checkpoint_is_accepted(tmp_path):
    jd = tmp_path / "jd"
    jd.mkdir()
    st = _state(1.0, 1)
    c0, c1 = _cap(jd, "run0", "0.pt", st, sha="aa"), _cap(jd, "run1", "0.pt", st, sha="aa")
    trusted.append_run(jd, dict(run_id="run0", start_step=0, resume_from=None, captures=[c0]))
    rec = dict(run_id="run1", start_step=1, resume_from="ck1", resume_sha256="aa", captures=[c1])
    trusted.append_run(jd, rec)
    assert reexec._transitions(jd, [rec], {("run1", c1["file"]): st}, {}, b"")[0] is True


# --- launcher I/O ---------------------------------------------------------------------------------------------
def test_sending_to_a_job_that_never_reads_neither_blocks_nor_grows_without_bound():
    import socket
    a, b = socket.socketpair()
    a.setblocking(False)
    rk = launcher._Rank(a)
    t0 = time.monotonic()
    for _ in range(20000):
        rk.send({"ids": list(range(64))})
    assert time.monotonic() - t0 < 5          # blocking sendall would hang here forever
    assert rk.overflow and len(rk.out) <= launcher.MAX_OUT_BYTES
    a.close(); b.close()


def test_a_peer_that_stops_reading_is_declared_stuck_after_the_deadline():
    import socket
    a, b = socket.socketpair()
    a.setblocking(False)
    rk = launcher._Rank(a)
    while not rk.out and not rk.overflow:       # fill the kernel buffer until something stays queued
        rk.send({"pad": "x" * 4096})
    now = time.monotonic()
    assert not rk.stuck(now) and rk.stuck(now + launcher.SEND_DEADLINE_S + 1)
    a.close(); b.close()


def test_a_slow_but_reading_peer_receives_everything_in_order():
    import socket
    a, b = socket.socketpair()
    a.setblocking(False); b.setblocking(False)
    rk = launcher._Rank(a)
    msgs = [{"n": i, "pad": "y" * 2000} for i in range(300)]
    for m in msgs:
        rk.send(m)
    got = b""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (rk.out or got.count(b"\n") < len(msgs)):
        rk.flush()
        try:
            got += b.recv(65536)
        except BlockingIOError:
            pass
    assert not rk.overflow and [__import__("json").loads(x)["n"] for x in got.splitlines()] == list(range(300))
    a.close(); b.close()
