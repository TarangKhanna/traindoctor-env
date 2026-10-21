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

# Rung 2: what the fixtures do. outcome_table replays every menu candidate (2 workers); the warm-up thread builds two
# scenarios' tables at once, so up to 4 jobs run together.
import os  # noqa: E402
from traindoctor_env.bundle import make_menu, outcome_table  # noqa: E402


def table(n):
    snap = snaps[n]
    menu = make_menu(snap, load_truth(snap))
    t0 = time.time()
    tab = outcome_table(snap, menu)
    c = next(m for m in menu if m.role == "correct")
    o = tab[c.id]
    bad = sum(not tab[m.id].ok for m in menu if m.role == "correct")
    return (f"{n}: correct ok={o.ok} steps={o.steps_completed} wall={time.time() - t0:.0f}s load={os.getloadavg()[0]:.1f} "
            f"n_menu={len(menu)} detail={o.detail!r} bad_correct={bad}")


for mode, workers in (("table_seq", 1), ("table_par2", 2)):
    with ThreadPoolExecutor(workers) as ex:
        for line in ex.map(table, NAMES):
            print(f"::notice title={mode}::{line}", flush=True)
