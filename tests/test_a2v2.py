"""Task A2 scenario version 2 (jobs/a2v2.py). Rules under test, not outputs: version 1 is untouched; the stall comes from
a producer that is slow to publish, not from a deletable wait; every case diff applies; the public data does not name the
fault; (executed) producer-side repairs recover with exact sample accounting while consumer-side wait removal does not."""
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

from jobs import a2v2
from jobs.build import EDITABLE, KINDS, LABELS, TEMPLATE, Edit, apply_edits, materialize
from jobs.config import loads
from traindoctor_env.scenario import Scenario, job_spec
from traindoctor_env.snapshot import job_files

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("redteam_a2v2", ROOT / "scripts" / "redteam_a2v2.py")
rr = importlib.util.module_from_spec(spec)
sys.modules["redteam_a2v2"] = rr
spec.loader.exec_module(rr)

SC = rr.SCENARIO


def test_version_1_is_untouched():
    """A2v2 is opt-in: not drawn by the generator, and v1 jobs materialize with the original data module and shards."""
    assert "A2v2" not in KINDS
    sp = job_spec(Scenario(101, "A2", 0, 3, 31337))
    assert not a2v2.is_v2(sp.config)
    with tempfile.TemporaryDirectory() as d:
        materialize(Path(d), sp.config, sp.bug)
        assert (Path(d) / "job" / "data.py").read_bytes() == (TEMPLATE / "data.py").read_bytes()
        assert (Path(d) / "data" / "part-0" / "shard0.idx").exists()


def test_v2_same_label_and_same_ranks_as_v1_for_the_same_seed():
    v1, v2 = job_spec(Scenario(101, "A2", 0, 3, 31337)), job_spec(SC)
    assert LABELS["A2v2"] == LABELS["A2"]
    assert (v1.details["rank"], v1.details["step"]) == (v2.details["rank"], v2.details["step"])


def test_the_bug_is_one_producer_rate_and_the_fix_is_its_reverse():
    sp = job_spec(SC)
    assert a2v2.is_v2(sp.config)
    assert len(sp.bug) == 1 and sp.bug[0].path == "job/config.yaml"
    with tempfile.TemporaryDirectory() as d:
        materialize(Path(d), sp.config, sp.bug)
        cfg = loads((Path(d) / "job" / "config.yaml").read_text())
        rank = str(sp.details["rank"])
        rates = cfg["data"]["producer"]["throttle_mb_s"]
        lag = lambda r: cfg["data"]["producer"]["shard_mb"] / rates[r]  # noqa: E731
        assert lag(rank) > 10 * cfg["pg_timeout_s"]                     # the injected producer cannot finish in time
        assert all(lag(r) < cfg["pg_timeout_s"] for r in rates if r != rank)   # every other producer, incl. the slow decoy, can
        apply_edits(Path(d), sp.fix)
        assert loads((Path(d) / "job" / "config.yaml").read_text()) == sp.config   # the repair restores the pristine config


def test_no_data_shards_are_precreated_the_producer_publishes_them():
    sp = job_spec(SC)
    with tempfile.TemporaryDirectory() as d:
        materialize(Path(d), sp.config, sp.bug)
        assert not (Path(d) / "data").exists() or not any((Path(d) / "data").rglob("*.idx"))
        src = (Path(d) / "job" / "data.py").read_text()
        assert "start_producer" in src and ".ready" in src


@pytest.fixture(scope="module")
def cases():
    sp = job_spec(SC)
    with tempfile.TemporaryDirectory() as d:
        materialize(Path(d), sp.config, sp.bug)
        files = job_files(Path(d))
        return files, rr.build_cases(files, sp.details, sp.fix)


def test_every_case_diff_applies_to_the_incident_workspace_and_touches_only_editable_files(cases, tmp_path):
    files, cs = cases
    sp = job_spec(SC)
    ids = [c.id for c in cs]
    assert {"null", "gold", "alt1", "alt2", "S01", "F06", "F11", "X2"} <= set(ids) and len(ids) == len(set(ids))
    for c in cs:
        ws = tmp_path / c.id
        materialize(ws, sp.config, sp.bug)
        apply_edits(ws, c.edits)
        assert {e.path for e in c.edits} <= set(EDITABLE), c.id


def test_expectations_are_the_spec_ones(cases):
    _, cs = cases
    by = {c.id: c for c in cs}
    assert by["null"].expected_ok is False and by["gold"].expected_ok and by["alt1"].expected_ok and by["alt2"].expected_ok
    assert by["S01"].expected_ok is False       # unsafe wait removal must be rejected
    assert all(not by[i].expected_ok for i in ("F06", "F11", "X2", "R1"))


def test_public_data_does_not_name_the_fault():
    """Ground truth stays out of the agent-visible material: no label, scenario kind, injected rate or fix text."""
    sp = job_spec(SC)
    with tempfile.TemporaryDirectory() as d:
        materialize(Path(d), sp.config, sp.bug)
        src = "".join(f for p in (Path(d) / "job").glob("*.py") for f in p.read_text())
    for needle in ("A2v2", "RANK_STALLED", "stalled", "injected", str(a2v2.STALLED_MB_S)):
        assert needle not in src, needle


# ---- executed (real jobs, local, unsandboxed: only the invariant fields are inspected, never `ok`) -----------------
@pytest.fixture(scope="module")
def snap(tmp_path_factory):
    from traindoctor_env.snapshot import execute
    return execute(SC, tmp_path_factory.mktemp("a2v2"))


def _outcome(snap, edits):
    from traindoctor_env.reexec import Replay
    rp = Replay(snap, allow_unsandboxed=True)
    try:
        for e in edits:
            rp.edit(e)
        return rp.outcome()
    finally:
        rp.close()


@pytest.mark.executed
def test_incident_is_a_stall_before_the_collective_with_a_collective_timeout_on_the_peers(snap):
    jobs = [r["text"] for r in snap.records if r["kind"] == "job"]
    assert len(jobs) == 1 and "error=COLLECTIVE_TIMEOUT" in jobs[0]
    rank = snap.reference and __import__("traindoctor_env.snapshot", fromlist=["load_truth"]).load_truth(snap).details["rank"]
    waits = [r for r in snap.records if r["kind"] == "log" and r["rank"] == rank and "waiting for shard" in r["text"]]
    assert waits
    assert not any("loaded shard 1" in r["text"] for r in snap.records if r["rank"] == rank)


@pytest.mark.executed
def test_producer_side_repairs_recover_with_exact_sample_accounting(snap):
    from traindoctor_env.snapshot import load_truth
    t = load_truth(snap)
    _, cs = (None, rr.build_cases(snap.files, t.details, [Edit(**e) for e in t.fix]))
    for c in cs:
        if c.id in ("gold", "alt1", "alt2"):
            o = _outcome(snap, c.edits)
            assert o.samples_match_reference and o.params_match_reference and o.ckpt_valid, c.id


@pytest.mark.executed
def test_removing_the_wait_reads_the_placeholder_and_breaks_sample_accounting(snap):
    """Rule: a rank that does not wait for its shard trains on batches the schedule did not assign it."""
    from traindoctor_env.snapshot import load_truth
    t = load_truth(snap)
    by = {c.id: c for c in rr.build_cases(snap.files, t.details, [Edit(**e) for e in t.fix])}
    for cid in ("S01", "R1"):
        o = _outcome(snap, by[cid].edits)
        assert not o.samples_match_reference and not o.params_match_reference and (o.sample_gaps or o.sample_dups), cid
