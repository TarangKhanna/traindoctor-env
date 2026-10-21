"""Diagnostic only (not a test): repeat oracle-repair verification under the OS sandbox and, for any outcome that is not
ok, keep the protocol directory and dump the ranks' stderr tails and the launcher's job record to diag.txt."""
import glob
import subprocess
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.conftest import NAMES, SCEN  # noqa: E402
from traindoctor_env import reexec  # noqa: E402
from traindoctor_env.oracle import reference_solution  # noqa: E402
from traindoctor_env.snapshot import execute, load_truth  # noqa: E402

KEEP = Path(tempfile.mkdtemp(prefix="keep"))


class Kept:
    """TemporaryDirectory stand-in that never deletes, so failed protocol directories can be read afterwards."""
    def __init__(self, *a, **k):
        self.name = tempfile.mkdtemp(dir=KEEP)

    def __enter__(self):
        return self.name

    def __exit__(self, *a):
        return False

    def cleanup(self):
        pass


reexec.tempfile.TemporaryDirectory = Kept
root = Path(tempfile.mkdtemp())
snaps = {n: execute(SCEN[n], root / n) for n in NAMES}
out = open("diag.txt", "w")
fails = 0
for rep in range(int(os.environ.get("REPS", "4"))):
    for n in NAMES:
        snap = snaps[n]
        _, _, edits = reference_solution(snap, load_truth(snap))
        before = set(KEEP.iterdir())
        t0 = time.time()
        o = reexec.verify_fix(snap, edits, allow_unsandboxed=False)
        line = f"rep{rep} {n}: ok={o.ok} steps={o.steps_completed} wall={time.time() - t0:.1f}s load={os.getloadavg()[0]:.1f} detail={o.detail!r}"
        out.write(line + "\n")
        if not o.ok:
            fails += 1
            log = subprocess.run(["log", "show", "--last", "2m", "--style", "compact", "--predicate", 'sender == "Sandbox"'],
                                 capture_output=True, text=True).stdout
            for x in sorted({l.split("deny", 1)[1][:160] for l in log.splitlines() if "deny" in l and "network" in l}):
                out.write(f"  DENY{x}\n")
            for d in sorted(set(KEEP.iterdir()) - before):
                for f in sorted(glob.glob(f"{d}/**/stderr/*.txt", recursive=True)):
                    out.write(f"  --- {Path(f).relative_to(d)}\n" + Path(f).read_text(errors="replace")[-1500:] + "\n")
                for f in sorted(glob.glob(f"{d}/**/job.json", recursive=True)):
                    j = json.loads(Path(f).read_text())
                    out.write(f"  --- {Path(f).relative_to(d)} status={j.get('status')} error_class={j.get('error_class')} "
                              f"ranks={[(r.get('rank'), r.get('returncode'), r.get('exc_class')) for r in j.get('ranks', [])]}\n")
        out.flush()
print(f"::notice title=diag::{fails} not-ok of {len(NAMES) * int(os.environ.get('REPS', '4'))}", flush=True)
