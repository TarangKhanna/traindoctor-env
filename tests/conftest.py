"""Executed scenarios are expensive (real Gloo jobs), so each is executed twice at most per session (in parallel,
to double as a determinism stress) and shared by all tests. At most two jobs run at once, and a background thread
warms the shared fixtures while the cheap tests run."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

os.environ.setdefault("TD_KEY", "test-suite-key")  # keys the data server's salt; tests never touch ~/.traindoctor

import jobs.resubmit
from jobs import sandbox
from traindoctor_env.bundle import make_menu, outcome_table
from traindoctor_env.scenario import Scenario
from traindoctor_env.snapshot import execute, load_truth

# A1/A2 share (seed, world, job_seed): the build streams are identical, so they form counterfactual pairs.
SCEN = {
    "A1_src": Scenario(101, "A1", 0, 2, 31337),
    "A1_cfg": Scenario(101, "A1", 1, 2, 31337),
    "A2": Scenario(101, "A2", 0, 2, 31337),
    "B_SKIP": Scenario(202, "B_SKIP", 0, 2, 777),
    "B_DUP": Scenario(202, "B_DUP", 0, 2, 777),
    "B_OK": Scenario(202, "B_OK", 0, 2, 777),
}
NAMES = list(SCEN)
MAX_JOBS = 2


class Cache:
    """Thread-safe compute-once cache."""

    def __init__(self):
        self.values, self.locks, self.guard = {}, {}, threading.Lock()

    def get(self, key, fn):
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self.values:
                self.values[key] = fn()
            return self.values[key]


@pytest.fixture(scope="session", autouse=True)
def limit_concurrent_jobs():
    slots, original = threading.BoundedSemaphore(MAX_JOBS), jobs.resubmit.run_job

    def limited(*args, **kwargs):
        with slots:
            return original(*args, **kwargs)

    jobs.resubmit.run_job = limited
    yield
    jobs.resubmit.run_job = original


@pytest.fixture(scope="session")
def snapshot_pair(tmp_path_factory):
    cache = Cache()

    def compute(name):
        base = tmp_path_factory.mktemp(name)
        with ThreadPoolExecutor(2) as ex:
            return tuple(ex.map(lambda n: execute(SCEN[name], base / n), ("a", "b")))

    return lambda name: cache.get(name, lambda: compute(name))


@pytest.fixture(scope="session")
def snapshot(snapshot_pair):
    return lambda name: snapshot_pair(name)[0]


@pytest.fixture
def needs_sandbox():
    """Tests that assert a legitimate repair is verified (ok=True). reexec.mark_sandbox forces ok=False for any
    unsandboxed re-execution, so on a runner without OS isolation (Linux CI) these cannot pass and skip instead."""
    if not sandbox.available():
        pytest.skip("sandbox-exec cannot be applied here; a verified outcome needs OS isolation")


@pytest.fixture(scope="session")
def menu_table(snapshot):
    """name -> (menu, truth, table of real re-execution outcomes)."""
    cache = Cache()

    if not sandbox.available():
        # Rule: a verified outcome needs OS isolation. Where it cannot be applied (Linux CI) the table is not built
        # with an unsandboxed opt-out (that would not be a verification); dependent tests skip instead.
        pytest.skip("sandbox-exec cannot be applied here; verified outcome tables need OS isolation")

    def compute(name):
        snap = snapshot(name)
        truth = load_truth(snap)
        menu = make_menu(snap, truth)
        return menu, truth, outcome_table(snap, menu)

    return lambda name: cache.get(name, lambda: compute(name))


@pytest.fixture(scope="session", autouse=True)
def warm_up(request, limit_concurrent_jobs):
    """Compute the shared snapshots and outcome tables in the background while cheap tests run."""
    if not any({"snapshot", "menu_table"} & set(item.fixturenames) for item in request.session.items):
        yield
        return
    request.getfixturevalue("tmp_path_factory").getbasetemp()  # create it before any thread asks
    pair = request.getfixturevalue("snapshot_pair")
    table = request.getfixturevalue("menu_table") if sandbox.available() else (lambda name: None)
    stop = threading.Event()

    def run():
        for name in NAMES:
            if not stop.is_set():
                pair(name), table(name)

    t = threading.Thread(target=run)
    t.start()
    yield
    stop.set()
    t.join()
