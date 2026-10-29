"""One separate-verifier Docker adapter; root launcher, unprivileged training ranks."""
from __future__ import annotations

import json
import errno
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from jobs import sandbox, trusted
from jobs.dataserver import SaltSchedule
from jobs.resubmit import resubmit
from traindoctor_env import reexec
from traindoctor_env.snapshot import reference_for_config, run_names

UID = 65534
MODE = "harbor_docker_uid"
PREFIX = ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups",
          "--no-new-privs", "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all"]
PROBE = r'''
import json, os, pathlib, socket, time
root = pathlib.Path(__import__('sys').argv[1])
s = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines())
result = {'uid': os.getuid(), 'gid': os.getgid(), 'groups': os.getgroups(),
          'uids': [int(v) for v in s['Uid'].split()], 'gids': [int(v) for v in s['Gid'].split()],
          'caps': {k: s[k].strip() for k in ('CapInh','CapEff','CapPrm','CapBnd','CapAmb')},
          'no_new_privs': s['NoNewPrivs'].strip()}
for name, action in [('private_read_denied', lambda: (root/'secret').read_text()),
                     ('private_write_denied', lambda: (root/'forged').write_text('forged'))]:
    try: action(); result[name] = False
    except PermissionError: result[name] = True
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
    connection.settimeout(0.5)
    started = time.monotonic()
    try:
        connection.connect(('198.51.100.1', 9))
        result['external_connect_errno'] = 0
    except OSError as exc:
        result['external_connect_errno'] = exc.errno
    result['external_connect_wall_s'] = time.monotonic() - started
print(json.dumps(result))
'''


def worker_pids():
    result = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in path.read_text().splitlines())
            if int(fields["Uid"].split()[0]) == UID and not fields["State"].strip().startswith("Z"):
                result.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, KeyError):
            pass
    return result


def stop_workers():
    # Only invoked after Docker/root/fresh-reserved-UID checks succeeded.
    for pid in worker_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2
    while worker_pids() and time.monotonic() < deadline:
        time.sleep(0.02)
    if worker_pids():
        raise sandbox.SandboxUnavailable("reserved worker processes did not terminate")


def root_owned(path):
    path = Path(path)
    for item in [path, *path.parents]:
        info = item.lstat()
        sticky_ancestor = item != path and stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISVTX
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or (info.st_mode & 0o022 and not sticky_ancestor):
            raise sandbox.SandboxUnavailable("trusted path is not root-owned and non-writable: " + str(item))


def plain_job_tree(job_dir):
    # No workers are alive during this walk/chown. Never chown a linked outside inode.
    from jobs import safefs
    safefs.listdir(job_dir)
    for directory, dirs, files in os.walk(job_dir, followlinks=False):
        for name in dirs + files:
            info = (Path(directory) / name).lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)) or info.st_nlink != 1 and stat.S_ISREG(info.st_mode):
                raise sandbox.SandboxUnavailable("job contains a link or special file before ownership change")


def prove_launcher_kill():
    script = "import os,time; print(os.getuid(),flush=True); time.sleep(3)"
    process = subprocess.Popen([*PREFIX, sys.executable, "-I", "-c", script],
                               stdout=subprocess.PIPE, text=True)
    try:
        if not select.select([process.stdout], [], [], 5)[0] or process.stdout.readline().strip() != str(UID):
            raise sandbox.SandboxUnavailable("kill probe did not enter worker uid")
        process.kill()
        if process.wait(timeout=5) != -signal.SIGKILL:
            raise sandbox.SandboxUnavailable("launcher cannot kill the worker uid")
    finally:
        process.wait(timeout=5)  # trusted probe exits itself if root lacks CAP_KILL
        process.stdout.close()


def preflight(fixture):
    if sys.platform != "linux" or os.geteuid() != 0 or not Path("/.dockerenv").is_file():
        raise sandbox.SandboxUnavailable("Harbor worker bridge requires a fresh root Docker verifier")
    if {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise sandbox.SandboxUnavailable("verifier must have network_mode:none (loopback only)")
    if Path("/proc/sys/fs/protected_hardlinks").read_text().strip() != "1":
        raise sandbox.SandboxUnavailable("kernel hardlink protection is required before job ownership changes")
    if worker_pids():
        raise sandbox.SandboxUnavailable("reserved worker uid already has running processes")
    for path in (fixture, Path(reexec.__file__), Path(sandbox.RUNTIME), Path(__file__), Path(sys.executable).resolve()):
        root_owned(path)
    with tempfile.TemporaryDirectory(prefix="td-proof-") as name:
        private = Path(name)
        (private / "secret").write_text("root-only probe")
        probe = subprocess.run([*PREFIX, sys.executable, "-I", "-c", PROBE, name],
                               check=True, capture_output=True, text=True, timeout=10)
        proof = json.loads(probe.stdout)
    valid = proof["uids"] == proof["gids"] == [UID] * 4 and proof["groups"] == []
    valid &= all(int(v, 16) == 0 for v in proof["caps"].values()) and proof["no_new_privs"] == "1"
    if proof["external_connect_errno"] not in {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EACCES, errno.EPERM}:
        raise sandbox.SandboxUnavailable("worker external network denial failed: errno="
            + str(proof["external_connect_errno"]) + ", elapsed_s=" + str(proof["external_connect_wall_s"]))
    if not valid or not proof["private_read_denied"] or not proof["private_write_denied"]:
        raise sandbox.SandboxUnavailable("worker privilege/private-file probe failed")
    prove_launcher_kill()
    return dict(proof, launcher_can_kill_worker=True, network_interfaces=["lo"], protected_hardlinks=1, exec_mode=MODE)


def export_evidence(jd, output, reference, schedule, proof, timings):
    root = trusted.trusted_dir(jd)
    count = total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        count += 1
        total += info.st_size if stat.S_ISREG(info.st_mode) else 0
        plain = stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        if count > 4096 or total > (512 << 20) or not plain:
            raise sandbox.SandboxUnavailable("trusted evidence export exceeds the artifact bound")
    evidence = output / "execution_evidence"
    evidence.mkdir(mode=0o700)
    shutil.copytree(root, evidence / "protocol.trusted")  # launcher-owned only; never copy the agent tree
    metadata = {"source": "executed", "exec_mode": MODE, "reference": reference,
                "schedule": {"base": schedule.base.hex(), "fresh": schedule.fresh.hex(), "boundary": schedule.boundary},
                "worker_isolation": proof, "timings": timings}
    (evidence / "replay.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def evaluate_replay(replay, fixture, output):
    started = time.perf_counter()
    proof = preflight(fixture)
    timings = {"isolation_preflight_wall_s": time.perf_counter() - started}
    old_wrap, calls = sandbox.wrap, []

    def wrap(argv, job_dir, allow_unsandboxed=False):
        if allow_unsandboxed:
            raise sandbox.SandboxUnavailable("unsandboxed fallback forbidden")
        stop_workers()  # remove children left by a preceding run before root filesystem work
        root = trusted.trusted_dir(job_dir)
        root.mkdir(exist_ok=True)
        root_owned(root)
        root.chmod(0o700)
        plain_job_tree(job_dir)
        subprocess.run(["chown", "-R", "-h", "-P", "65534:65534", str(job_dir)], check=True)
        calls.append(str(job_dir))
        return [*PREFIX, *argv]

    sandbox.wrap = wrap
    try:
        with tempfile.TemporaryDirectory(prefix="td-harbor-") as name:
            directory = Path(name)
            directory.chmod(0o711)  # traverse to the rank-owned job; sibling ledger stays root-only
            jd, before, frontier, kw, tampered = reexec._fresh_protocol_dir(replay.snapshot, replay.jd, directory)
            schedule = SaltSchedule.fresh_for(replay.snapshot.salt, 0)
            started = time.perf_counter()
            reference = reference_for_config(replay.snapshot.reference["pristine_config"], schedule)
            timings["trusted_reference_wall_s"] = time.perf_counter() - started
            started = time.perf_counter()
            result = resubmit(jd, sandbox=True, allow_unsandboxed=False, salt=schedule, **kw)
            timings["recovery_execution_wall_s"] = time.perf_counter() - started
            stop_workers()
            records = [r for r in trusted.ledger(jd) if r["run_id"] not in before]
            if len(calls) != len(records) or not records or not all(r["sandboxed"] for r in records):
                raise sandbox.SandboxUnavailable("execution did not use the proved worker wrapper")
            started = time.perf_counter()
            outcome = reexec.check(jd, replay.snapshot, reexec.lost_work(jd, frontier, before),
                tampered, before, expect_kill=True, relaunches=len(run_names(jd)) - len(before),
                ref=reference, salt=schedule, format_check="v1.2", refusal=reexec.refusal_detail(result))
            timings["semantic_verification_wall_s"] = time.perf_counter() - started
            outcome.exec_mode = MODE  # actual wrapper and evidence above; no gVisor/Modal claim
            export_evidence(jd, output, reference, schedule, proof, timings)
            return dict(asdict(outcome), worker_isolation=proof, timings=timings)
    finally:
        sandbox.wrap = old_wrap
        stop_workers()
