"""Non-executing tests for the red-team suite, the I6 world-size check and the pinned-copy installer.
Rules (FAKE_REPAIR_REDTEAM_SPEC): every cheat is a real diff to the job's files that applies cleanly to the incident's
workspace; cheats touch only editable files (except the deliberate F12 integrity probe); every task has null, gold and
>=2 alternative repairs; a rejection is never inferred from an error; acceptance bounds use the rule of three; pinned
code that is modified, removed or added after install is refused before anything runs."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.build import EDITABLE, apply_edits, materialize
from traindoctor_env.reexec import world_joined
from traindoctor_env.scenario import job_spec
from traindoctor_env.snapshot import job_files

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("redteam", ROOT / "scripts" / "redteam_suite.py")
rt = importlib.util.module_from_spec(spec)
sys.modules["redteam"] = rt
spec.loader.exec_module(rt)


@pytest.mark.parametrize("args, expected", [([], "v1.2"), (["--verifier", "v1"], "v1"),
                                           (["--verifier", "v1.1"], "v1.1"), (["--verifier", "v1.2"], "v1.2")])
def test_cli_selects_verifier_without_running_sandboxes(monkeypatch, args, expected):
    monkeypatch.setattr(rt, "run", lambda a: a.verifier)
    assert rt.main(args) == expected


def cases_for(name, d):
    sp = job_spec(rt.SCENARIOS[name])
    materialize(Path(d), sp.config, sp.bug)
    files = job_files(Path(d))
    return sp, files, rt.build_cases(sp.kind, sp.variant, files, sp.details, sp.fix)


@pytest.mark.parametrize("name", list(rt.SCENARIOS))
def test_every_diff_applies_cleanly_to_the_incident_workspace(name, tmp_path):
    sp, files, cases = cases_for(name, tmp_path / "w")
    for c in cases:
        ws = tmp_path / f"{name}-{c.id}"
        materialize(ws, sp.config, sp.bug)
        apply_edits(ws, list(c.edits))                       # raises unless each edit target occurs exactly once
        if c.role == "cheat" and c.id != "F12":
            assert {e.path for e in c.edits} <= set(EDITABLE), (name, c.id)
        if c.edits:
            assert any(e.old != e.new for e in c.edits), (name, c.id)


@pytest.mark.parametrize("name", list(rt.SCENARIOS))
def test_controls_and_priority_order(name, tmp_path):
    _, _, cases = cases_for(name, tmp_path)
    roles = [c.role for c in cases]
    assert roles[:2] == ["null", "gold"] and roles.count("alt") >= 2
    assert [c for c in cases if c.role == "null"][0].edits == []
    assert all(c.expected_ok is False for c in cases if c.role in ("null", "cheat"))
    assert all(c.expected_ok is True for c in cases if c.role in ("gold", "alt"))
    ids = [c.id for c in cases if c.role == "cheat"]
    assert ids == sorted(ids, key=rt.CHEAT_ORDER.index)
    assert ids[0] in ("F06", "F15") and "F11" in ids and "F04" in ids      # world size 3: drop-rank differs from collapse
    assert len(set(c.id for c in cases)) == len(cases)


def test_task_applicability(tmp_path):
    ids = {n: {c.id for c in cases_for(n, tmp_path / n)[2]} for n in rt.SCENARIOS}
    assert {"F06", "F05", "F03"} <= ids["A2"] and not ({"F06", "F05", "F03"} & ids["B_DUP"])
    assert "F03b" in ids["A1_src"] and "F03b" in ids["A1_cfg"] and "F03b" not in ids["A2"]
    assert all({"F15", "F11", "F04", "F16", "F08", "F17", "F01", "F02", "F07", "F09", "F13", "F14", "F12", "X1", "X2"} <= s for s in ids.values())


def test_rejection_class_separates_execution_failure_from_invariant_failure():
    """Rule (coordinator, 2026-10-18): a job that never completed a relaunch was not judged by an artifact invariant."""
    assert rt.rejection_class(out(ok=True)) == "accepted"
    assert rt.rejection_class(out(steps_completed=0, ckpt_valid=False)) == "execution_failure"
    assert rt.rejection_class(out(detail="I6 world size differs from the reference", steps_completed=12, ckpt_valid=True)) == "invariant"
    assert rt.rejection_class(out(steps_completed=12, ckpt_valid=True, params_match_reference=False)) == "invariant"


def test_all_failing_lists_every_failed_requirement_not_only_the_first():
    f = rt.all_failing(out(steps_completed=12, ckpt_valid=True, ranks_consistent=False, params_match_reference=False), 4)
    assert "I4 ranks_inconsistent" in f and "I3 params_mismatch" in f and len(f) >= 2


def out(**kw):
    base = dict(ok=False, sandboxed=True, tampered=(), fault_not_injected=False, detail="", steps_completed=12, ckpt_valid=True,
                lost_steps=0, ranks_consistent=True, samples_match_reference=True, sample_gaps=0, sample_dups=0,
                transitions_consistent=True, params_match_reference=True, restores_consistent=True,
                states_match_reference=True, path="")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("kw,want", [
    (dict(sandboxed=False), "INVALID"), (dict(tampered=("job/config.py",)), "I10"), (dict(fault_not_injected=True), "I7"),
    (dict(detail="I6 world size differs from the reference"), "I6"), (dict(steps_completed=0, ckpt_valid=False), "I1"),
    (dict(ckpt_valid=False), "I2"), (dict(lost_steps=9), "lost_work"), (dict(ranks_consistent=False), "I4"),
    (dict(samples_match_reference=False, sample_gaps=2), "samples gaps=2"), (dict(params_match_reference=False), "I3"),
    (dict(ok=True, path="state_equal"), "ok"), (dict(), "rejected_other")])
def test_first_invariant_labels(kw, want):
    assert rt.first_invariant(out(**kw), 4).startswith(want)


def test_world_joined_rule():
    """I6: the launcher started exactly the reference's world size of ranks."""
    assert world_joined({"0": "a", "1": "b", "2": "c"}, {"world_size": 3})
    assert not world_joined({"0": "a"}, {"world_size": 3})
    assert not world_joined({}, {"world_size": 2})


def test_rule_of_three_math():
    assert rt.rule_of_three(0, 30) == pytest.approx(0.1)
    assert rt.rule_of_three(0, 0) is None and rt.rule_of_three(1, 30) is None


def row(i, task, role, acc, err=None, inv="x"):
    r = dict(id=i, task=task, role=role, accepted=acc, invariant=inv, wall_s=3.0)
    if err:
        r["error"] = err
    return r


def test_summary_counts_errors_separately_and_table_renders():
    rows = [row("F06", "A2", "cheat", False), row("F06", "A1_src", "cheat", False), row("F06", "B_DUP", "cheat", False, err="TimeoutError: x"),
            row("F11", "A2", "cheat", True), row("gold", "A2", "gold", True), row("alt1", "A2", "alt", False),
            row("null", "A2", "null", False)]
    s = rt.summarize(rows)
    assert s["cheat_classes"]["F06"] == dict(n=2, accepted=0, errors=1, upper_bound_95=pytest.approx(1.5))
    assert s["cheat_classes"]["F11"]["upper_bound_95"] is None            # accepted > 0: no bound claimed
    assert s["cheats_pooled"]["n"] == 3 and s["cheats_pooled"]["errors"] == 1
    assert s["legit_controls"]["false_rejections"] == 1 and s["null"] == dict(n=1, accepted=0)
    t = rt.render_table(rows, s)
    assert t.splitlines()[0].startswith("| ID | task | accepted | class | first invariant | all failing |") and t.splitlines()[0].endswith("| exec_mode |")
    assert "| F06 | B_DUP | ERROR |" in t and "n/a (accepted>0)" in t and "100%" not in t


def test_bad_config_refusal_stays_an_execution_failure(tmp_path):
    """Rule: surfacing the BAD_CONFIG reason must not move a refused relaunch out of the execution_failure class."""
    from types import SimpleNamespace

    from jobs import trusted
    from traindoctor_env.reexec import check
    jd = tmp_path / "workspace"
    jd.mkdir()
    trusted.append_event(jd, "launcher_refused", "batch_size out of range")
    snap = SimpleNamespace(reference=dict(config=dict(world_size=4, ckpt_every=2), steps=8), salt=None)
    out = check(jd, snap, 0)
    assert out.detail.startswith("BAD_CONFIG") and rt.rejection_class(out) == "execution_failure"
