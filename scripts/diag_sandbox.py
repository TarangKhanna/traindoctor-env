"""Diagnose where the macOS sandbox profile in jobs/sandbox.py stops a PyTorch/Gloo process.

Runs a ladder of small commands under the real profile and records exit code and the tail of stdout and stderr for
each rung, then the Sandbox denials the system logged. It changes nothing about the profile and bypasses nothing.

    python scripts/diag_sandbox.py [--denials]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobs import sandbox  # noqa: E402

PORT = 29577
RANKS = """
import os, sys, torch, torch.distributed as d
r = int(sys.argv[1]); w = int(sys.argv[2])
d.init_process_group('gloo', init_method='tcp://127.0.0.1:%s' % sys.argv[3], rank=r, world_size=w)
t = torch.ones(1) * (r + 1)
d.all_reduce(t)
open('rank%d.txt' % r, 'w').write(str(t.item()))
print('rank', r, 'sum', t.item())
d.destroy_process_group()
"""

LADDER = [
    ("L0 python starts", ["-c", "print(1)"]),
    ("L1 import torch", ["-c", "import torch; print(torch.__version__)"]),
    ("L2 torch.distributed gloo available", ["-c", "import torch.distributed as d; print(d.is_gloo_available())"]),
    ("L3 write and read a file in the run dir", ["-c", "open('x.txt','w').write('a'); print(open('x.txt').read())"]),
    ("L4 torch.save and load in the run dir", ["-c", "import torch; torch.save({'a': torch.ones(2)}, 'c.pt'); print(torch.load('c.pt', weights_only=True)['a'].sum().item())"]),
    ("L5 one-rank gloo all_reduce", ["-c", RANKS, "0", "1", str(PORT)]),
]


def run_sandboxed(run_dir: Path, args: list[str], env_extra: dict | None = None, timeout: int = 90):
    argv = sandbox.wrap([sys.executable, *args], run_dir)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    return subprocess.Popen(argv, cwd=run_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def tail(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    return s[-n:]


def main() -> int:
    out = {"sandbox_available": sandbox.available(), "python": sys.executable, "rungs": []}
    if not out["sandbox_available"]:
        print(json.dumps(out))
        return 0
    with tempfile.TemporaryDirectory(prefix="tdsb-") as td:
        run_dir = Path(td).resolve()
        for name, args in LADDER:
            p = run_sandboxed(run_dir, args)
            try:
                o, e = p.communicate(timeout=90)
                out["rungs"].append(dict(name=name, rc=p.returncode, stdout=tail(o), stderr=tail(e)))
            except subprocess.TimeoutExpired:
                p.kill()
                o, e = p.communicate()
                out["rungs"].append(dict(name=name, rc="timeout", stdout=tail(o), stderr=tail(e)))
            if out["rungs"][-1]["rc"] != 0:
                break
        else:
            # two ranks, each sandboxed, rendezvous over loopback
            procs = [run_sandboxed(run_dir, ["-c", RANKS, str(r), "2", str(PORT + 1)]) for r in range(2)]
            res = []
            for r, p in enumerate(procs):
                try:
                    o, e = p.communicate(timeout=90)
                    res.append(dict(rank=r, rc=p.returncode, stdout=tail(o), stderr=tail(e)))
                except subprocess.TimeoutExpired:
                    p.kill()
                    o, e = p.communicate()
                    res.append(dict(rank=r, rc="timeout", stdout=tail(o), stderr=tail(e)))
            out["rungs"].append(dict(name="L6 two-rank gloo all_reduce", ranks=res))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
