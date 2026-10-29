"""Adapter boundary tests only; mocked outcomes do not establish Harbor execution."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "harbor/tasks/training-repair-01/tests/grade.py"
SPEC = importlib.util.spec_from_file_location("harbor_task_grade", PATH)
grade = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grade)


def submission(tmp_path, source="x = 1\n"):
    root = tmp_path / "submission"
    root.mkdir()
    (root / "ckpt.py").write_text(source)
    return root


def test_unsupported_backend_never_runs_submitted_code_or_issues_reward(monkeypatch, tmp_path):
    marker = tmp_path / "host-executed"
    root = submission(tmp_path, f"open({str(marker)!r}, 'w').write('unsafe')\n")
    monkeypatch.setattr(grade.sys, "platform", "darwin")
    monkeypatch.setattr(grade.sandbox, "available", lambda: False)
    monkeypatch.setattr(grade, "Replay", lambda *a, **k: pytest.fail("must not launch without isolation"))
    out = tmp_path / "verifier"
    out.mkdir()
    (out / "reward.json").write_text('{"reward": 1}')
    assert grade.run(root, out, tmp_path / "fixture") == 2
    result = json.loads((out / "grade_result.json").read_text())
    assert result["status"] == "infrastructure_error" and result["terminal_reward"] is None
    assert not result["execution_performed"] and not (out / "reward.json").exists()
    assert not marker.exists()


@pytest.mark.parametrize("bad", ["symlink", "hardlink", "extra", "oversize", "syntax", "root_symlink"])
def test_artifact_boundary_rejects_invalid_edits_before_execution(monkeypatch, tmp_path, bad):
    root = submission(tmp_path)
    path = root / "ckpt.py"
    if bad == "symlink":
        path.unlink(); path.symlink_to(tmp_path / "outside")
    elif bad == "hardlink":
        (tmp_path / "outside").hardlink_to(path)
    elif bad == "extra":
        (root / "ledger.jsonl").write_text("forged")
    elif bad == "oversize":
        path.write_text("#" * (grade.MAX_SOURCE_BYTES + 1))
    elif bad == "syntax":
        path.write_text("def broken(:")
    else:
        link = tmp_path / "root-link"; link.symlink_to(root, target_is_directory=True); root = link
    monkeypatch.setattr(grade, "evaluate", lambda *a: pytest.fail("invalid artifact must not execute"))
    out = tmp_path / "verifier"
    assert grade.run(root, out, tmp_path / "fixture") == 0
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0}
    assert json.loads((out / "grade_result.json").read_text())["status"] == "invalid_submission"


@pytest.mark.parametrize("ok", [False, True])
def test_reward_uses_trusted_semantic_verdict_only(monkeypatch, tmp_path, ok):
    root = submission(tmp_path)
    outcome = {"ok": ok, "sandboxed": True, "transitions_consistent": ok}
    monkeypatch.setattr(grade, "evaluate", lambda *a: outcome)
    out = tmp_path / "verifier"
    assert grade.run(root, out, tmp_path / "fixture") == 0
    assert json.loads((out / "reward.json").read_text())["reward"] == int(ok)
    result = json.loads((out / "grade_result.json").read_text())
    assert result["verdict"] == outcome and result["status"] == "graded"


def test_linux_evaluation_rejects_an_unconfined_backend_result(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade.Snapshot, "load", lambda _: object())
    closed = []
    monkeypatch.setattr(grade, "Replay", lambda *a, **k: SimpleNamespace(
        write_file=lambda *a: True, close=lambda: closed.append(True)))
    monkeypatch.setattr(grade, "evaluate_replay", lambda *a: {"ok": True, "sandboxed": False})
    out = tmp_path / "verifier"
    assert grade.run(submission(tmp_path), out, tmp_path / "fixture") == 2
    assert closed == [True] and not (out / "reward.json").exists()


def test_linux_bridge_cannot_activate_on_the_mac_host(monkeypatch, tmp_path):
    import worker_bridge
    monkeypatch.setattr(worker_bridge.sys, "platform", "darwin")
    monkeypatch.setattr(worker_bridge, "worker_pids", lambda: pytest.fail("must not inspect/kill host workers"))
    with pytest.raises(grade.sandbox.SandboxUnavailable, match="fresh root Docker"):
        worker_bridge.preflight(tmp_path)


def test_execution_failure_restores_global_wrapper(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import worker_bridge
    original = worker_bridge.sandbox.wrap
    monkeypatch.setattr(worker_bridge, "preflight", lambda _: {"proof": "simulated unit test"})
    monkeypatch.setattr(worker_bridge, "stop_workers", lambda: None)
    monkeypatch.setattr(worker_bridge.reexec, "_fresh_protocol_dir",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("injected preparation failure")))
    with pytest.raises(RuntimeError, match="preparation failure"):
        worker_bridge.evaluate_replay(SimpleNamespace(snapshot=None, jd=None), tmp_path, tmp_path)
    assert worker_bridge.sandbox.wrap is original


@pytest.mark.parametrize("bad", ["hardlink", "symlink", "fifo"])
def test_worker_tree_rejects_linked_inodes_before_root_chown(tmp_path, bad):
    import os
    import worker_bridge
    root = tmp_path / "job"
    root.mkdir()
    target = root / "artifact"
    outside = tmp_path / "outside"
    outside.write_text("must remain owned by the trusted side")
    before = outside.stat()
    if bad == "hardlink":
        target.hardlink_to(outside)
    elif bad == "symlink":
        target.symlink_to(outside)
    else:
        os.mkfifo(target)
    with pytest.raises(grade.sandbox.SandboxUnavailable, match="link or special"):
        worker_bridge.plain_job_tree(root)
    assert (outside.stat().st_uid, outside.stat().st_gid) == (before.st_uid, before.st_gid)


def test_evidence_export_copies_only_launcher_owned_state(tmp_path):
    from types import SimpleNamespace
    import worker_bridge
    job = tmp_path / "protocol"
    job.mkdir()
    (job / "agent_file").write_text("must not be exported")
    private = worker_bridge.trusted.trusted_dir(job)
    private.mkdir()
    (private / "ledger.jsonl").write_text('{"simulated_unit_test": true}\n')
    output = tmp_path / "output"
    output.mkdir()
    schedule = SimpleNamespace(base=b"base", fresh=b"fresh", boundary=0)
    worker_bridge.export_evidence(job, output, {"simulated": True}, schedule, {}, {})
    evidence = output / "execution_evidence"
    assert (evidence / "protocol.trusted/ledger.jsonl").is_file()
    assert not list(evidence.rglob("agent_file"))
    assert json.loads((evidence / "replay.json").read_text())["schedule"]["fresh"] == b"fresh".hex()


@pytest.mark.parametrize("bad", ["symlink", "oversize"])
def test_evidence_export_refuses_unsafe_or_oversize_files(tmp_path, bad):
    import worker_bridge
    private = worker_bridge.trusted.trusted_dir(tmp_path / "protocol")
    private.mkdir()
    target = private / "capture"
    if bad == "symlink":
        target.symlink_to(tmp_path / "outside")
    else:
        with target.open("wb") as stream:
            stream.truncate((512 << 20) + 1)  # sparse file: no large allocation
    with pytest.raises(grade.sandbox.SandboxUnavailable, match="artifact bound"):
        worker_bridge.export_evidence(tmp_path / "protocol", tmp_path, {}, None, {}, {})
    assert not (tmp_path / "execution_evidence").exists()
