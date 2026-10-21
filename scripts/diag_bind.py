"""Diagnostic only: try binding a TCP socket to each lo0 address under the real job sandbox profile."""
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobs import sandbox  # noqa: E402

addrs = sorted({a[4][0] for fam in (socket.AF_INET, socket.AF_INET6)
                for a in socket.getaddrinfo("localhost", None, fam)} | {"127.0.0.1", "::1", "fe80::1%lo0"})
print("::notice title=lo0::" + subprocess.run(["ifconfig", "lo0"], capture_output=True, text=True).stdout.replace("\n", " | ")[:400])
run = Path(tempfile.mkdtemp())
prof = sandbox.profile(run)
for a in addrs:
    fam = "AF_INET6" if ":" in a else "AF_INET"
    code = f"import socket; s=socket.socket(socket.{fam}); s.bind(({a!r}, 0)); s.listen(); print('bound', s.getsockname())"
    for label, argv in (("sandboxed", ["sandbox-exec", "-p", prof, sys.executable, "-c", code]),
                        ("unsandboxed", [sys.executable, "-c", code])):
        r = subprocess.run(argv, capture_output=True, text=True, cwd=run)
        msg = (r.stdout or r.stderr).strip().splitlines()[-1][-160:]
        print(f"::notice title=bind {a} {label}::rc={r.returncode} {msg}")
