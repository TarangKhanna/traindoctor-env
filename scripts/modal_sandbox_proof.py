"""Proof that the Modal Sandbox backend (modal_train/sandbox_backend.py) can verify untrusted repairs.
Step 1: network is blocked and loopback works inside the sandbox; gold repair on a Task A variant is ACCEPTED with
exec_mode=modal_sandbox. Step 2: fake repairs F06, F11 are REJECTED (reason named); gold re-run is deterministic.
Writes modal_train/results/modal_sandbox_proof.json. Cost is booked via modal_train/budget.py (cpu_per_hour)."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import os  # noqa: E402

try:
    import modal  # noqa: F401
except ImportError:   # the Modal client lives in a separate venv; appended AFTER the repo venv so torch etc. win
    sys.path.append(os.environ["TD_MODAL_SITE"])
from modal_train import budget, sandbox_backend  # noqa: E402

OUT = ROOT / "modal_train" / "results" / "modal_sandbox_proof.json"
NET_PROBE = r'''
import json, socket, urllib.request
res = {}
for name, fn in (("https_example_com", lambda: urllib.request.urlopen("https://example.com", timeout=8).status),
                 ("dns_example_com", lambda: socket.gethostbyname("example.com"))):
    try:
        res[name] = f"REACHED: {fn()}"
    except Exception as e:
        res[name] = f"FAILED: {type(e).__name__}: {e}"
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1); port = s.getsockname()[1]
c = socket.create_connection(("127.0.0.1", port), timeout=5); a, _ = s.accept(); c.sendall(b"ok")
res["loopback"] = "OK" if a.recv(2) == b"ok" else "BAD"
import torch, torch.distributed as dist, os
res["torch"] = torch.__version__
print(json.dumps(res))
'''


def spend(label: str, seconds: float) -> None:
    budget.record(f"proof-{label}-{int(time.time())}", "modal_sandbox_proof", "cpu_per_hour", seconds * 4, "actual")  # 4 cpu


def network_probe() -> dict:
    import modal
    app = modal.App.lookup(sandbox_backend.APP_NAME, create_if_missing=True)
    t0 = time.time()
    sb = modal.Sandbox.create(image=sandbox_backend.image(), app=app, block_network=True, cpu=1, memory=1024, timeout=60, secrets=[])
    try:
        sb.filesystem.write_text(NET_PROBE, "/tmp/p.py")
        p = sb.exec("python", "/tmp/p.py")
        rc = p.wait()
        out, err = p.stdout.read(), p.stderr.read()
    finally:
        sb.terminate()
    spend("net", time.time() - t0)
    return dict(rc=rc, result=json.loads(out.strip().splitlines()[-1]) if rc == 0 else None, stderr=err[-800:], wall_s=time.time() - t0)


def main() -> int:
    from jobs.build import Edit  # noqa: F401
    from redteam_suite import SCENARIOS, build_cases, first_invariant
    from traindoctor_env import reexec
    from traindoctor_env.oracle import reference_solution
    from traindoctor_env.snapshot import execute, load_truth
    assert not reexec.DEV_CONTAINER_MODE
    t_all = time.time()
    res: dict = dict(steps={})
    net = network_probe()
    res["steps"]["network_probe"] = net
    r = net["result"] or {}
    net_ok = r.get("https_example_com", "").startswith("FAILED") and r.get("dns_example_com", "").startswith("FAILED") and r.get("loopback") == "OK"
    print("network probe:", json.dumps(net), "->", "PASS" if net_ok else "FAIL", flush=True)
    if not net_ok:
        res["verdict"] = "network/loopback probe failed"
        OUT.write_text(json.dumps(res, indent=1)); return 1
    name = sys.argv[1] if len(sys.argv) > 1 else "A2"
    sc = SCENARIOS[name]
    rows = []
    with tempfile.TemporaryDirectory() as wd:
        snap = execute(sc, Path(wd) / name)       # trusted, local, unsandboxed: as in prepare
        truth = load_truth(snap)
        _, _, gold = reference_solution(snap, truth)
        cases = {c.id: c for c in build_cases(sc.kind, sc.variant, snap.files, truth.details, gold)}
        ckpt_every = snap.reference["config"]["ckpt_every"]
        for cid in ["gold", "F06", "F11", "gold"]:
            t0 = time.time()
            row = dict(id=cid, run=len(rows) + 1)
            try:
                rp = reexec.Replay(snap, allow_unsandboxed=False, backend="modal")
                try:
                    for e in cases[cid].edits:
                        rp.edit(e)
                    o = rp.outcome()
                    row["backend_info"] = rp.backend_info
                finally:
                    rp.close()
                row.update(accepted=bool(o.ok and o.sandboxed), exec_mode=o.exec_mode, sandboxed=o.sandboxed,
                           invariant=first_invariant(o, ckpt_every), detail=o.detail,
                           outcome={k: (list(v) if isinstance(v, tuple) else v) for k, v in o.__dict__.items()})
            except BaseException as e:  # noqa: BLE001 -- recorded, fail closed
                row.update(accepted=False, error=f"{type(e).__name__}: {e}")
            row["wall_s"] = time.time() - t0
            spend(cid, row["wall_s"])
            rows.append(row)
            print(f"{cid:5} accepted={row['accepted']} mode={row.get('exec_mode')} {row.get('error') or row['invariant']} ({row['wall_s']:.0f}s)", flush=True)
            OUT.write_text(json.dumps(dict(res, task=name, rows=rows), indent=1, default=str))
    g1, g2 = rows[0], rows[3]
    checks = dict(
        gold_accepted_modal=bool(g1["accepted"] and g1.get("exec_mode") == "modal_sandbox"),
        F06_rejected=not rows[1]["accepted"] and not rows[1].get("error"),
        F11_rejected=not rows[2]["accepted"] and not rows[2].get("error"),
        gold_deterministic=(g1["accepted"] == g2["accepted"] and g1.get("invariant") == g2.get("invariant")
                            and g1.get("outcome", {}).get("steps_completed") == g2.get("outcome", {}).get("steps_completed")))
    res.update(task=name, rows=rows, checks=checks, total_wall_s=time.time() - t_all,
               ledger_total_usd=budget.spent_usd())
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(checks), f"total {res['total_wall_s']:.0f}s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
