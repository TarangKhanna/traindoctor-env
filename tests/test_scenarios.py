"""Task A and Task B scenarios: reproducible, indistinguishable where they must be, oracle repairs pass, wrong
fixes fail on the invariant that matters."""
import json

import pytest

from tests.conftest import NAMES, SCEN
from traindoctor_env.oracle import reference_solution
from traindoctor_env.snapshot import evidence_ids, load_truth

pytestmark = pytest.mark.executed
PUBLIC_FILES = ["records.jsonl", "config.json", "manifest.json", "files.json"]


def texts(snap, kind, run=None):
    return [r["text"] for r in snap.records if r["kind"] == kind and (run is None or r["text"].startswith(f"[{run}]"))]


@pytest.mark.parametrize("name", NAMES)
def test_reproducible_from_seed_and_the_incident_really_happened(name, snapshot_pair):
    a, b = snapshot_pair(name)
    for f in PUBLIC_FILES:
        assert (a.public_dir / f).read_bytes() == (b.public_dir / f).read_bytes(), f
    jobs = texts(a, "job")
    assert "failed" in jobs[0] and "error=COLLECTIVE_TIMEOUT" in jobs[0]
    if SCEN[name].kind.startswith("B"):  # worker lost, scheduler restarted from a checkpoint, run completed
        assert len(jobs) == 2 and "-> completed" in jobs[1] and "resume_from=ck" in jobs[1]
        assert any("killed" in t for t in texts(a, "process", "run0"))
    else:                                # no restart yet; peers timed out
        assert len(jobs) == 1


@pytest.mark.parametrize("a1", ["A1_src", "A1_cfg"])
def test_a1_and_a2_share_the_job_level_symptom_but_not_the_evidence(a1, snapshot):
    """Counterfactual pair: same canonical job status and the same peer errors; only the traces/logs differ."""
    s1, s2 = snapshot(a1), snapshot("A2")
    assert texts(s1, "job") == texts(s2, "job")
    rank = load_truth(s1).details["rank"]
    assert rank == load_truth(s2).details["rank"]
    peers = lambda s: sorted(t for r in s.records if r["kind"] == "process" and r["rank"] != rank for t in [r["text"]])  # noqa: E731
    assert peers(s1) == peers(s2) and all("COLLECTIVE_TIMEOUT" in t for t in peers(s1))
    # distinguishing evidence: A1's divergent rank entered another op; A2's stalled rank never did and waits on a shard
    ops = lambda s: {r["text"].split()[5] for r in s.records if r["kind"] == "trace" and r["rank"] == rank}  # noqa: E731
    assert ops(s1) - {"all_reduce"} and ops(s2) == {"all_reduce"}
    waits = lambda s: [r for r in s.records if r["kind"] == "log" and r["rank"] == rank and "waiting for shard" in r["text"]]  # noqa: E731
    assert not waits(s1) and waits(s2)
    stalled = [t for r in s2.records if r["kind"] == "process" and r["rank"] == rank for t in [r["text"]]]
    assert "killed" in stalled[0]  # the stalled rank never errors: the watchdog kills it
    seq = lambda s: max(int(r["text"].split("seq=")[1].split()[0]) for r in s.records if r["kind"] == "trace" and r["rank"] == rank)  # noqa: E731
    pending_peer = lambda s: min(int(r["text"].split("seq=")[1].split()[0]) for r in s.records  # noqa: E731
                                 if r["kind"] == "trace" and r["rank"] != rank and "pending" in r["text"])
    assert seq(s2) < pending_peer(s2)      # A2: the stalled rank never entered the collective its peers wait in
    assert seq(s1) >= pending_peer(s1)     # A1: it entered one (a different op)


@pytest.mark.parametrize("name", NAMES)
def test_oracle_repair_passes_every_invariant_by_reexecution(name, menu_table, snapshot):
    menu, truth, table = menu_table(name)
    _, evidence, edits = reference_solution(snapshot(name), truth)
    correct = [c for c in menu if c.role == "correct"]
    assert len(correct) == 1 and evidence      # the menu shows the repair in its one-line form; the table decides
    out = table[correct[0].id]
    steps = snapshot(name).config["steps"]
    assert out.ok and out.steps_completed == steps and out.params_match_reference and out.samples_match_reference
    assert out.no_duplicate_updates and out.ranks_consistent and out.sample_gaps == out.sample_dups == 0
    assert out.lost_steps <= snapshot(name).config["ckpt_every"]


# wrong fix -> the invariant that must be flagged
FLAGGED = {
    "noop": lambda o: not o.ok,
    "decoy": lambda o: not o.ok,                                            # a wrong edit at a location that holds no cause
    "exclude_rank": lambda o: not o.ok and not o.params_match_reference and not o.samples_match_reference,
}


@pytest.mark.parametrize("name", NAMES)
def test_wrong_fixes_fail_with_the_right_invariant_flagged(name, menu_table):
    menu, truth, table = menu_table(name)
    seen = set()
    for c in menu:
        if c.role == "correct":
            assert table[c.id].ok, c.role   # an alternative repair that really fixes the job passes too
            continue
        seen.add(c.role)
        assert FLAGGED["decoy" if c.role.startswith("decoy") else c.role](table[c.id]), (c.role, table[c.id])
    if truth.kind in ("B_SKIP", "B_DUP"):   # skipped / replayed samples are caught by committed sample ids
        noop = next(table[c.id] for c in menu if c.role == "noop")
        assert noop.sample_gaps > 0 and noop.sample_dups > 0 and noop.no_duplicate_updates
    if truth.kind == "B_OK":                # a control with nothing to repair: the relaunch is the correct answer
        assert table[next(c.id for c in menu if c.role == "correct")].ok
    assert seen


def test_blanket_restart_fails_task_b(menu_table):
    """Task B: 'resume again' (no change) completes all steps but leaves the sample history wrong."""
    for name in ("B_SKIP", "B_DUP"):
        menu, _, table = menu_table(name)
        o = table[next(c.id for c in menu if c.role == "noop")]
        assert not o.ok and not o.samples_match_reference and o.steps_completed == 12


def test_evidence_ids_are_opaque_across_kinds(snapshot):
    """Same seed, different mechanism: ids at the same record index are identical, so they cannot encode the cause."""
    a, b = snapshot("A1_src"), snapshot("A2")
    n = min(len(a.records), len(b.records))
    assert [r["id"] for r in a.records][:n] == [r["id"] for r in b.records][:n] == evidence_ids(101, n)
    assert {r["source"] for r in a.records} == {"executed"} and len({r["id"] for r in a.records}) == len(a.records)
    assert json.loads((a.public_dir / "config.json").read_text())["seed"] == 31337 != a.seed
