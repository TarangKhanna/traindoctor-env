"""Diagnostic only (not a test): verify each scenario's oracle repair under the OS sandbox, first sequentially, then two
at a time as the test fixtures do, and print the outcome fields that the executed tests do not show."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.conftest import NAMES, SCEN  # noqa: E402
from traindoctor_env.oracle import reference_solution  # noqa: E402
from traindoctor_env.reexec import verify_fix  # noqa: E402
from traindoctor_env.snapshot import execute, load_truth  # noqa: E402

root = Path(tempfile.mkdtemp())
snaps = {n: execute(SCEN[n], root / n) for n in NAMES}


def one(n):
    snap = snaps[n]
    _, _, edits = reference_solution(snap, load_truth(snap))
    t0 = time.time()
    o = verify_fix(snap, edits, allow_unsandboxed=False)
    return (f"{n}: ok={o.ok} steps={o.steps_completed}/{snap.config['steps']} relaunch={o.relaunches} "
            f"gaps={o.sample_gaps} dups={o.sample_dups} params={o.params_match_reference} lost={o.lost_steps} "
            f"sandboxed={o.sandboxed} wall={time.time() - t0:.0f}s detail={o.detail!r}")


for mode, workers in (("sequential", 1), ("parallel2", 2)):
    with ThreadPoolExecutor(workers) as ex:
        for line in ex.map(one, NAMES):
            print(f"::notice title={mode}::{line}", flush=True)
