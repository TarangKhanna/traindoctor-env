"""Rules (no job launched): trusted files are never named or located by job-supplied strings; trusted run records
live outside job-writable paths; the CI opt-out is honored only on Linux CI and never yields verified success."""
import json
import os
from types import SimpleNamespace

import pytest

from jobs import trusted
from jobs.trusted import Capturer, trusted_dir
from traindoctor_env import verifiers
from traindoctor_env.reexec import FixOutcome, ci_unsandboxed_allowed, mark_sandbox


def _job(tmp_path, entries, files=None):
    jd = tmp_path / "job"
    (jd / "ckpt").mkdir(parents=True)
    for rel, data in (files or {}).items():
        (jd / "ckpt" / rel).parent.mkdir(parents=True, exist_ok=True)
        (jd / "ckpt" / rel).write_bytes(data)
    (jd / "ckpt" / "MANIFEST.json").write_text(json.dumps(entries))
    return jd


def _all_files(root):
    return {p for p in root.rglob("*") if p.is_file()}


def test_honest_manifest_is_captured_under_counter_and_hash_names(tmp_path):
    jd = _job(tmp_path, [dict(ckpt_id="ck0001", step=1, shards=[dict(file="ck0001/s.pt")])], {"ck0001/s.pt": b"abc"})
    c = Capturer(jd, "run0"); c.scan()
    assert len(c.records) == 1 and c.records[0]["ckpt_id"] == "ck0001"
    assert "ck0001" not in c.records[0]["file"]
    assert (trusted_dir(jd) / c.records[0]["file"]).read_bytes() == b"abc"


@pytest.mark.parametrize("cid", ["../../evil", "/abs/evil", "a/../../b"])
def test_job_supplied_ckpt_id_never_reaches_a_path(tmp_path, cid):
    jd = _job(tmp_path, [dict(ckpt_id=cid, step=1, shards=[dict(file="s.pt")])], {"s.pt": b"x"})
    before = _all_files(tmp_path)
    c = Capturer(jd, "run0"); c.scan()
    new = _all_files(tmp_path) - before
    assert new and all(trusted_dir(jd).resolve() in p.resolve().parents for p in new)
    assert c.records[0]["ckpt_id"] == cid          # kept as data only


@pytest.mark.parametrize("rel", ["../x", "/etc/passwd", "a/../../x", "", "./s.pt"])
def test_shard_path_escaping_job_ckpt_dir_is_rejected(tmp_path, rel):
    (tmp_path / "x").write_bytes(b"secret")
    jd = _job(tmp_path, [dict(ckpt_id="c", step=1, shards=[dict(file=rel)])])
    c = Capturer(jd, "run0"); c.scan()
    assert c.records == [] and c.invalid
    assert not trusted_dir(jd).exists() or not _all_files(trusted_dir(jd))


def test_symlinked_shard_or_dir_is_rejected(tmp_path):
    outside = tmp_path / "outside.pt"; outside.write_bytes(b"secret")
    (tmp_path / "outdir").mkdir(); (tmp_path / "outdir" / "s.pt").write_bytes(b"secret")
    jd = _job(tmp_path, [dict(ckpt_id="c", step=1, shards=[dict(file="link.pt")]),
                         dict(ckpt_id="d", step=2, shards=[dict(file="dlink/s.pt")])])
    os.symlink(outside, jd / "ckpt" / "link.pt")
    os.symlink(tmp_path / "outdir", jd / "ckpt" / "dlink")
    c = Capturer(jd, "run0"); c.scan()
    assert c.records == [] and len(c.invalid) == 2


def test_run_records_go_to_trusted_dir_and_never_follow_planted_symlinks(tmp_path):
    jd = tmp_path / "job"
    run = jd / "runs" / "run0"; run.mkdir(parents=True)
    victim = tmp_path / "victim"; victim.mkdir()
    os.symlink(victim, run / "samples")
    os.symlink(victim / "job.json", run / "job.json")
    p1 = trusted.write_run_record(jd, "run0", "samples/rank0.jsonl", "{}\n")
    p2 = trusted.write_run_record(jd, "run0", "job.json", "{}")
    assert not list(victim.iterdir())
    for p in (p1, p2):
        assert trusted_dir(jd).resolve() in p.resolve().parents
    assert trusted.read_run_record(jd, "run0", "job.json") == "{}"


@pytest.mark.parametrize("run_id,name", [("../run0", "job.json"), ("run0", "../../x"), ("run0", "/x"), ("evil", "job.json")])
def test_run_record_names_cannot_escape(tmp_path, run_id, name):
    with pytest.raises((AssertionError, ValueError)):
        trusted.write_run_record(tmp_path / "job", run_id, name, "x")


def test_read_nofollow_refuses_symlink_components(tmp_path):
    (tmp_path / "real").mkdir(); (tmp_path / "real" / "f").write_text("hi")
    base = tmp_path / "base"; base.mkdir()
    os.symlink(tmp_path / "real", base / "d")
    os.symlink(tmp_path / "real" / "f", base / "f")
    for rel in ("d/f", "f", "../real/f"):
        with pytest.raises((OSError, ValueError)):
            trusted.read_nofollow(base, rel)


def test_unsandboxed_optout_honored_only_on_linux_ci():
    on = {"TD_CI_UNSANDBOXED": "1", "CI": "true"}
    assert ci_unsandboxed_allowed(on, "linux")
    assert not ci_unsandboxed_allowed(on, "darwin")
    assert not ci_unsandboxed_allowed({"TD_CI_UNSANDBOXED": "1"}, "linux")
    assert not ci_unsandboxed_allowed({"CI": "true"}, "linux")


def test_unsandboxed_outcome_is_invalid_and_never_verified_success():
    out = FixOutcome(True, True, True, True, True, True, 0, True, path="state_equal")
    assert out.sandboxed is True
    mark_sandbox(out, False)
    assert out.sandboxed is False and out.ok is False and out.path == "" and "INVALID" in out.detail


def test_verdict_never_verified_when_unsandboxed(monkeypatch):
    monkeypatch.setattr(verifiers, "evidence_ok", lambda *a, **k: True)
    truth = SimpleNamespace(label="L")
    mk = lambda sb: FixOutcome(True, True, True, True, True, True, 0, True, path="state_equal", sandboxed=sb)  # noqa: E731
    assert verifiers.verdict(truth, "L", [], set(), [], mk(True)).v2
    assert not verifiers.verdict(truth, "L", [], set(), [], mk(False)).v2


def test_default_mark_sandbox_forces_ok_false_and_dev_container_never_passes_strict_v2(monkeypatch):
    """Rule: without isolation nothing verifies; DEV_CONTAINER_MODE keeps the real outcome but strict V2 still rejects it."""
    from traindoctor_env import reexec
    assert reexec.DEV_CONTAINER_MODE is False
    out = mark_sandbox(FixOutcome(True, True, True, True, True, True, 0, True, path="state_equal"), False)
    assert not out.ok and out.exec_mode == "sandboxed"
    dev = FixOutcome(True, True, True, True, True, True, 0, True, path="state_equal", sandboxed=False, exec_mode="dev_container")
    monkeypatch.setattr(verifiers, "evidence_ok", lambda *a, **k: True)
    assert not verifiers.verdict(SimpleNamespace(label="L"), "L", [], set(), [], dev).v2


def test_world_shrink_edit_yields_fault_not_injected_not_exception(tmp_path):
    """Rule: a planned worker failure on a rank the config lacks is never skipped silently nor an exception."""
    from jobs.build import Edit
    from traindoctor_env.reexec import verify_fix
    from traindoctor_env.scenario import make_scenario
    from traindoctor_env.snapshot import execute
    snap = execute(next(s for s in map(make_scenario, range(300)) if s.kind == "B_SKIP"), tmp_path / "s")
    rank, w = snap.meta["kill"][0], snap.config["world_size"]
    out = verify_fix(snap, [Edit("job/config.yaml", f"world_size: {w}", f"world_size: {rank}")], allow_unsandboxed=True)
    assert out.fault_not_injected and not out.ok and "fault plan not carried out" in out.detail


def _bare_snapshot():
    return SimpleNamespace(reference=dict(config=dict(world_size=4, ckpt_every=2), steps=8), salt=None)


def test_refused_config_reports_bad_config_reason_and_never_verifies(tmp_path):
    """Rule: a relaunch the launcher refused is a non-completion, and the outcome says why (BAD_CONFIG + reason)."""
    from traindoctor_env.reexec import check
    jd = tmp_path / "workspace"
    jd.mkdir()
    trusted.append_event(jd, "resubmit_refused", "world_size out of range")
    out = check(jd, _bare_snapshot(), 0)
    assert not out.ok and out.steps_completed == 0 and out.path == ""
    assert out.detail == "BAD_CONFIG: world_size out of range"


def test_no_refusal_keeps_empty_detail(tmp_path):
    """Rule: the detail names BAD_CONFIG only when the launcher actually refused; otherwise the old empty detail stays."""
    from traindoctor_env.reexec import check
    jd = tmp_path / "workspace"
    jd.mkdir()
    assert check(jd, _bare_snapshot(), 0).detail == ""
    trusted.append_event(jd, "relaunch", "resume ckpt_000004")
    assert check(jd, _bare_snapshot(), 0).detail == ""



def test_incomplete_relaunch_reports_launcher_status_only(tmp_path):
    """Rule: a relaunch that did not complete is a non-completion whose detail carries the launcher's own status."""
    from traindoctor_env.reexec import check
    jd = tmp_path / "workspace"
    jd.mkdir()
    trusted.append_run(jd, dict(run_id="r1", status="watchdog_killed", completed=False, captures=[]))
    out = check(jd, _bare_snapshot(), 0)
    assert not out.ok and out.path == "" and out.detail == "NOT_COMPLETED: last relaunch watchdog_killed"
