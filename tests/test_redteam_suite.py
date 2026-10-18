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


# --- pinned-copy installer --------------------------------------------------------------------------------------
INSTALLER = ROOT.parent.parent / "tools" / "apply_td_sandbox_specs.py"
needs_installer = pytest.mark.skipif(not INSTALLER.exists(), reason="installer lives in the main checkout")


@pytest.fixture
def installed(tmp_path):
    wt, d = tmp_path / "wt", tmp_path / "tdrun"
    for rel, body in {"scripts/recheck_table_sandboxed.py": "print(1)\n", "scripts/redteam_suite.py": "print(2)\n",
                      "jobs/a.py": "A=1\n", "jobs/sub/b.py": "B=1\n", "traindoctor_env/c.py": "C=1\n", "rl/d.py": "D=1\n",
                      "modal_train/cache/dev_eps.pkl": "dev", "modal_train/cache/train_eps.pkl": "train"}.items():
        (wt / rel).parent.mkdir(parents=True, exist_ok=True)
        (wt / rel).write_text(body)
    d.mkdir()
    uv, venv = tmp_path / "uvroot" / "cpython-x" / "bin", tmp_path / "venv"
    uv.mkdir(parents=True)
    (uv / "python3.12").write_text("#!/bin/sh\n")
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(uv / "python3.12")
    (venv / "pyvenv.cfg").write_text("home = x\n")
    (venv / "lib/python3.12/site-packages/pkg").mkdir(parents=True)
    (venv / "lib/python3.12/site-packages/pkg/__init__.py").write_text("X=1\n")
    (venv / "lib/python3.12/site-packages/easy.pth").write_text("pkg\n")
    (d / "registry.json").write_text(json.dumps({"specs": {"other": {"argv": ["x"]}}}))
    env = dict(os.environ, TDRUN_DIR=str(d), TD_WT=str(wt), TD_PY=str(venv / "bin" / "python"), TD_UV_ROOT=str(tmp_path / "uvroot"))
    r = subprocess.run([sys.executable, str(INSTALLER)], input="yes\n", env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return d, wt, r.stdout


def refuse(d, which="td_redteam_run.sh", *args):
    return subprocess.run(["/bin/bash", str(d / which), *args], capture_output=True, text=True)


@needs_installer
def test_installer_registers_specs_and_pins_a_copy(installed):
    d, wt, stdout = installed
    reg = json.loads((d / "registry.json").read_text())["specs"]
    assert {"other", "td_recheck_dev", "td_recheck_train", "td_redteam"} <= set(reg)
    rt_spec = reg["td_redteam"]
    assert rt_spec["profile"] == "trusted" and rt_spec["heavy"] and rt_spec["caffeinate"] and rt_spec["args_re"] is None
    assert rt_spec["argv"] == ["/bin/bash", str(d / "td_redteam_run.sh")]
    pins = json.loads((d / "td_recheck" / "td_pins.json").read_text())["files"]
    assert "jobs/sub/b.py" in pins and "modal_train/cache/dev_eps.pkl" in pins and len(pins) == 8
    assert "review" in stdout.lower() and "re-run" in stdout.lower()
    wrapper = (d / "td_redteam_run.sh").read_text()
    assert str(wt) not in wrapper.replace(str(wt / "modal_train" / "results"), "")   # worktree only as the results dir
    assert "-I -B" in wrapper and "PYTHONDONTWRITEBYTECODE=1" in wrapper and "unset" in wrapper and "PYTHONPATH" in wrapper
    ok = subprocess.run(["/usr/bin/python3", "-I", str(d / "td_pin_verify.py"), str(d / "td_recheck")], capture_output=True, text=True)
    assert ok.returncode == 0 and "pins ok" in ok.stdout


@needs_installer
@pytest.mark.parametrize("tamper,needle", [("code", "modified jobs/a.py"), ("pickle", "modified modal_train/cache/dev_eps.pkl"),
                                           ("extra", "extra file jobs/evil.py"), ("missing", "missing rl/d.py")])
def test_wrapper_refuses_tampered_copy(installed, tamper, needle):
    d, wt, _ = installed
    c = d / "td_recheck"
    if tamper == "code":
        os.chmod(c / "jobs/a.py", 0o600); (c / "jobs/a.py").write_text("A=2\n")
    elif tamper == "pickle":
        os.chmod(c / "modal_train/cache/dev_eps.pkl", 0o600); (c / "modal_train/cache/dev_eps.pkl").write_text("evil")
    elif tamper == "extra":
        (c / "jobs/evil.py").write_text("import os\n")
    else:
        (c / "rl/d.py").unlink()
    for wrapper, args in (("td_redteam_run.sh", ()), ("td_recheck_run.sh", ("dev",))):
        r = refuse(d, wrapper, *args)
        assert r.returncode == 78 and needle in r.stderr, (wrapper, r.returncode, r.stderr)


@needs_installer
def test_editing_the_worktree_after_install_changes_nothing_the_wrapper_runs(installed):
    d, wt, _ = installed
    (wt / "jobs/a.py").write_text("A=666\n")
    ok = subprocess.run(["/usr/bin/python3", "-I", str(d / "td_pin_verify.py"), str(d / "td_recheck")], capture_output=True, text=True)
    assert ok.returncode == 0                                            # the pinned copy is what runs
    assert (d / "td_recheck/jobs/a.py").read_text() == "A=1\n"


def test_driver_refuses_pickles_not_in_the_owner_manifest(tmp_path, monkeypatch):
    drv = importlib.util.spec_from_file_location("recheck2", ROOT / "scripts" / "recheck_table_sandboxed.py")
    m = importlib.util.module_from_spec(drv); drv.loader.exec_module(m)
    (tmp_path / "x_eps.pkl").write_bytes(b"data")
    monkeypatch.setattr(m, "CACHE", tmp_path)
    pins = tmp_path / "pins.json"
    monkeypatch.setenv("TD_PINS", str(pins))
    import hashlib
    pins.write_text(json.dumps({"files": {}}))
    with pytest.raises(SystemExit):
        m._check_provenance("x_eps.pkl")                                  # not in manifest (a PROVENANCE.md would not help)
    pins.write_text(json.dumps({"files": {"modal_train/cache/x_eps.pkl": "0" * 64}}))
    with pytest.raises(SystemExit):
        m._check_provenance("x_eps.pkl")
    pins.write_text(json.dumps({"files": {"modal_train/cache/x_eps.pkl": hashlib.sha256(b"data").hexdigest()}}))
    m._check_provenance("x_eps.pkl")


VENV_TAMPER = [("pth", "modified lib/python3.12/site-packages/easy.pth" if False else "extra file lib/python3.12/site-packages/evil.pth"),
               ("lib", "modified lib/python3.12/site-packages/pkg/__init__.py"),
               ("extra_pyc", "extra file lib/python3.12/site-packages/pkg/sitecustomize.py"),
               ("cfg", "modified pyvenv.cfg"), ("missing", "missing lib/python3.12/site-packages/easy.pth"),
               ("relink", "interpreter resolves elsewhere")]


@needs_installer
@pytest.mark.parametrize("tamper,needle", VENV_TAMPER)
def test_wrapper_refuses_a_tampered_venv(installed, tamper, needle):
    """A .pth / sitecustomize / edited library in the venv would run unsandboxed under the trusted profile."""
    d, wt, _ = installed
    venv = d.parent / "venv"
    sp = venv / "lib/python3.12/site-packages"
    if tamper == "pth":
        (sp / "evil.pth").write_text("import os\n")
    elif tamper == "lib":
        (sp / "pkg/__init__.py").write_text("X=2\n")
    elif tamper == "extra_pyc":
        (sp / "pkg/sitecustomize.py").write_text("import os\n")
    elif tamper == "cfg":
        (venv / "pyvenv.cfg").write_text("home = elsewhere\n")
    elif tamper == "missing":
        (sp / "easy.pth").unlink()
    else:
        other = d.parent / "other-python"
        other.write_text("#!/bin/sh\n")
        (venv / "bin" / "python").unlink()
        (venv / "bin" / "python").symlink_to(other)
    for wrapper, args in (("td_redteam_run.sh", ()), ("td_recheck_run.sh", ("train",))):
        r = refuse(d, wrapper, *args)
        assert r.returncode == 78 and needle in r.stderr, (wrapper, r.returncode, r.stderr)


@needs_installer
def test_installer_refuses_an_interpreter_outside_the_uv_root(tmp_path):
    wt, d = tmp_path / "wt", tmp_path / "tdrun"
    d.mkdir(); (d / "registry.json").write_text('{"specs": {}}')
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)          # not under TD_UV_ROOT
    env = dict(os.environ, TDRUN_DIR=str(d), TD_WT=str(wt), TD_PY=str(venv / "bin" / "python"), TD_UV_ROOT=str(tmp_path / "uvroot"))
    (tmp_path / "uvroot").mkdir()
    r = subprocess.run([sys.executable, str(INSTALLER)], input="yes\n", env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "outside the uv python root" in r.stderr
    assert not (d / "td_recheck").exists()


@needs_installer
def test_installer_upgrades_an_existing_install_after_showing_the_diff(installed):
    d, wt, _ = installed
    (wt / "jobs/a.py").write_text("A=2\n")                         # reviewed change: needs a re-install
    env = dict(os.environ, TDRUN_DIR=str(d), TD_WT=str(wt), TD_PY=str(d.parent / "venv/bin/python"), TD_UV_ROOT=str(d.parent / "uvroot"))
    declined = subprocess.run([sys.executable, str(INSTALLER)], input="no\n", env=env, capture_output=True, text=True)
    assert declined.returncode != 0 and "UPGRADE" in declined.stdout and "changed: ['jobs/a.py']" in declined.stdout
    assert (d / "td_recheck/jobs/a.py").read_text() == "A=1\n"      # nothing replaced without 'yes'
    r = subprocess.run([sys.executable, str(INSTALLER)], input="yes\n", env=env, capture_output=True, text=True)
    assert r.returncode == 0 and (d / "td_recheck/jobs/a.py").read_text() == "A=2\n"
    assert list(d.glob("td_recheck.bak-*")) and list(d.glob("registry.json.bak-*"))
    ok = subprocess.run(["/usr/bin/python3", "-I", str(d / "td_pin_verify.py"), str(d / "td_recheck")], capture_output=True, text=True)
    assert ok.returncode == 0 and "venv entries" in ok.stdout
