"""Replay backend for UNTRUSTED agent repairs: the Modal Sandbox API (modal.Sandbox.create, block_network=True, gVisor).
Used by traindoctor_env.reexec when Replay/verify_fix is given backend='modal'. Existing Modal functionality only: no
bespoke isolation. The host side stays trusted Python: it ships the prepared protocol directory (scenario snapshot +
ONLY the agent's diff to editable files, as built by reexec._fresh_protocol_dir) into the sandbox, runs the same
jobs.resubmit entry, and copies the produced directory back as DATA. Digests and invariants are recomputed on the
trusted side from the copied checkpoints (reexec.check); nothing the sandbox reports is trusted.

Inside the container the launcher runs as root and the job's rank processes run as uid 65534 (setpriv), so agent code
cannot read the launcher's salt (passed over stdin only) nor write the trusted ledger (root-only dir)."""
from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

EXEC_MODAL = "modal_sandbox"
ROOT = Path(__file__).resolve().parent.parent
TORCH_VERSION = "2.14.1"          # == pyproject.toml pin (asserted in tests)
NUMPY_VERSION = "2.5.4"           # == uv.lock (asserted in tests)
APP_NAME = "traindoctor-sandbox"
WORK = "/work"
MAX_BACK_BYTES = 512 << 20

ENTRY = r'''
import json, os, subprocess, sys
sys.path.insert(0, "/app")
from pathlib import Path
from jobs import sandbox
from jobs.dataserver import SaltSchedule
from jobs.resubmit import resubmit
args = json.loads(sys.stdin.read())
# launcher-made dirs (runs/<id>, ...) are root-owned: hand the job tree to the job uid just before each rank starts
sandbox.wrap = lambda argv, job_dir, allow: ["sh", "-c", 'chown -R -h 65534:65534 "$0"; exec setpriv --reuid=65534 --regid=65534 --clear-groups "$@"', str(job_dir), *argv]
jd = Path("/work/protocol")
subprocess.run(["chmod", "-R", "a+rwX", str(jd)], check=True)
Path(str(jd) + ".trusted").mkdir(exist_ok=True)   # Task B starts with no trusted dir; the launcher creates it as root
subprocess.run(["chmod", "-R", "go-rwx", str(jd) + ".trusted"], check=True)
sched = SaltSchedule(bytes.fromhex(args["base"]), bytes.fromhex(args["fresh"]), args["boundary"])
kw = args["kw"]
if kw.get("kill"):
    kw["kill"] = tuple(kw["kill"])
r = resubmit(jd, sandbox=True, salt=sched, **kw)
diag = {}
if not r.ok:   # diagnostics for the host log only; never an input to the verdict
    for f in sorted(jd.glob("runs/*/stderr/rank*.txt")):
        diag[str(f.relative_to(jd))] = f.read_text(errors="replace")[-600:]
print(json.dumps(dict(ok=r.ok, status=r.status, error_class=r.error_class,
                      message=r.message if r.status == "refused" else "", stderr=diag)))
'''


def entry_result(stdout: str) -> dict:
    """The entry's own result line (status, error_class, message) as plain strings, or {} if absent or malformed.
    Sandbox output is data: reexec uses it only to explain a relaunch the ledger shows never started."""
    try:
        d = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    return {k: d[k] for k in ("status", "error_class", "message") if isinstance(d.get(k), str)}


def spec_args(sched, kw: dict) -> dict:
    """JSON-safe form of (SaltSchedule, resubmit kwargs); pure, unit-tested."""
    kw = dict(kw)
    if kw.get("kill"):
        kw["kill"] = list(kw["kill"])
    return dict(base=sched.base.hex(), fresh=sched.fresh.hex(), boundary=int(sched.boundary), kw=kw)


def tar_dir(paths: dict[str, Path]) -> bytes:
    """tar of {arcname: directory}; regular files and dirs only (symlinks/devices never leave the host)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for arc, p in paths.items():
            tf.add(p, arcname=arc, filter=lambda ti: ti if (ti.isreg() or ti.isdir()) else None)
    return buf.getvalue()


def safe_extract(blob: bytes, dest: Path) -> int:
    """Extract an UNTRUSTED tar: regular files and directories only, no absolute or '..' names. Raises otherwise
    (fail closed: the verification errors out rather than reading a half-trusted tree)."""
    n = 0
    dest = Path(dest).resolve()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tf:
        for m in tf:
            if not (m.isreg() or m.isdir()):
                raise ValueError(f"sandbox output contains a non-regular member: {m.name!r}")
            target = (dest / m.name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"sandbox output member escapes the destination: {m.name!r}")
            n += m.size
            if n > MAX_BACK_BYTES:
                raise ValueError("sandbox output too large")
            m.mode, m.uid, m.gid, m.uname, m.gname = (0o755 if m.isdir() else 0o644), 0, 0, "", ""
            tf.extract(m, dest, filter="fully_trusted")   # members already vetted above
    return n


def image():
    import modal
    return (modal.Image.debian_slim(python_version="3.12")
            .env({"PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"})
            .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cpu")
            .pip_install(f"numpy=={NUMPY_VERSION}")
            .add_local_dir(ROOT / "jobs", "/app/jobs", copy=True, ignore=["__pycache__", "*.pyc"]))


_IMAGE = None


def run_resubmit(jd: Path, sched, kw: dict, timeout: int = 120, cpu: int = 4, memory: int = 4096) -> dict:
    """Run resubmit(jd, ...) for UNTRUSTED job code in a network-less gVisor Modal Sandbox and replace the contents of
    `jd` (and jd.trusted) with what came back. Returns timing facts. Raises on any transport/extraction problem."""
    import modal
    global _IMAGE
    jd = Path(jd)
    t0 = time.time()
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    _IMAGE = _IMAGE or image()
    trusted = jd.with_name(jd.name + ".trusted")
    payload = tar_dir({"protocol": jd, "protocol.trusted": trusted} if trusted.exists() else {"protocol": jd})
    sb = modal.Sandbox.create(image=_IMAGE, app=app, block_network=True, cpu=cpu, memory=memory, timeout=timeout,
                              secrets=[], workdir=WORK)
    info: dict = dict(sandbox_id=sb.object_id)
    try:
        sb.filesystem.write_bytes(payload, f"{WORK}/in.tar")
        sb.exec("tar", "-xf", f"{WORK}/in.tar", "-C", WORK).wait()
        sb.exec("rm", "-f", f"{WORK}/in.tar").wait()      # job uid could read the pre-incident trusted ledger from it
        sb.filesystem.write_text(ENTRY, "/app/entry.py")
        sb.exec("chmod", "600", "/app/entry.py").wait()
        p = sb.exec("python", "/app/entry.py", timeout=timeout)
        p.stdin.write(json.dumps(spec_args(sched, kw)))
        p.stdin.write_eof()
        p.stdin.drain()
        rc = p.wait()
        out = p.stdout.read()
        info.update(entry_rc=rc, entry_stdout=out[-4000:], entry_stderr=p.stderr.read()[-2000:], entry_result=entry_result(out))
        if rc != 0:
            raise RuntimeError(f"sandbox entry failed rc={rc}: {info['entry_stderr'][-500:]}")
        names = ["protocol", "protocol.trusted"]     # the entry always creates the trusted dir (Task B has none beforehand)
        sb.exec("tar", "-cf", f"{WORK}/out.tar", "-C", WORK, *names).wait()
        blob = sb.filesystem.read_bytes(f"{WORK}/out.tar")
    finally:
        sb.terminate()
    with tempfile.TemporaryDirectory() as tmp:
        safe_extract(blob, Path(tmp))
        for src, dst in ((Path(tmp) / "protocol", jd), (Path(tmp) / "protocol.trusted", trusted)):
            if src.exists():
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
    info.update(wall_s=time.time() - t0, out_bytes=len(blob))
    return info
