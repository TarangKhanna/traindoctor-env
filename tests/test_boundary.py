"""Rules for the execution boundary (Codex review an internal boundary review). The harness runs unconfined; the
job (its source, everything in its run dir / workspace) is hostile and controls timing. Unit-level: no job is launched."""
import hashlib
import json
import os
import re
from types import SimpleNamespace

import pytest
import torch

from jobs import manifest, resubmit as rs, sandbox, trusted
from jobs.dataserver import DataServer, DataSpec, salt_for
from jobs.trusted import Capturer, trusted_dir
from traindoctor_env import reexec, verifiers
from traindoctor_env.reexec import EDITABLE, Replay, _fresh_protocol_dir, check
from traindoctor_env.scenario import Truth


# ---- finding 1: the sandbox profile grants nothing outside the run dir -------------------------------------------
def test_profile_writes_only_inside_the_run_dir_and_shm_is_name_restricted(tmp_path):
    p = sandbox.profile(tmp_path)
    writes = [ln for ln in p.splitlines() if ln.startswith("(allow file-write")]
    assert len(writes) == 1 and "regex" not in writes[0]
    assert re.findall(r'\(subpath "([^"]+)"\)', writes[0]) == [str(tmp_path.resolve())]
    assert "KMP" not in writes[0] and not any("/tmp" in x for x in re.findall(r'"([^"]+)"', writes[0]) if x != str(tmp_path.resolve()))
    shm = [ln for ln in p.splitlines() if "ipc-posix-shm" in ln]
    assert shm and all("ipc-posix-name-prefix" in ln for ln in shm)         # never the unrestricted `(allow ipc-posix-shm)`


# ---- finding 2: trusted I/O into job-writable trees never follows a symlink --------------------------------------
def _victim(tmp_path):
    v = tmp_path / "victim"
    v.mkdir()
    (v / "secret").write_text("S3CRET")
    return v


def test_scheduler_events_are_trusted_storage_not_a_job_writable_file(tmp_path):
    victim = _victim(tmp_path)
    jd = tmp_path / "job"; jd.mkdir()
    os.symlink(victim / "secret", jd / "scheduler.jsonl")              # planted by the job before it exited
    rs.event(jd, "submitted", "run0")
    assert (victim / "secret").read_text() == "S3CRET"
    assert trusted.events(jd) == [dict(event="submitted", detail="run0")]
    assert trusted_dir(jd).resolve() in (trusted_dir(jd) / "scheduler.jsonl").resolve().parents


@pytest.mark.parametrize("plant", ["stderr_dir", "stderr_file", "runs_dir"])
def test_stderr_and_run_dirs_are_opened_without_following_planted_links(tmp_path, plant):
    from jobs import safefs
    victim = _victim(tmp_path)
    jd = tmp_path / "job"; (jd / "runs" / "run0").mkdir(parents=True)
    if plant == "stderr_dir":
        os.symlink(victim, jd / "runs" / "run0" / "stderr")
    elif plant == "stderr_file":
        (jd / "runs" / "run0" / "stderr").mkdir()
        os.symlink(victim / "secret", jd / "runs" / "run0" / "stderr" / "rank0.txt")
    else:
        (jd / "runs" / "run0").rmdir(); (jd / "runs").rmdir()
        os.symlink(victim, jd / "runs")
    with pytest.raises(OSError):
        safefs.makedirs(jd, "runs/run0/stderr")
        safefs.open_write(jd, "runs/run0/stderr/rank0.txt").close()
    assert sorted(p.name for p in victim.iterdir()) == ["secret"] and (victim / "secret").read_text() == "S3CRET"


def test_a_job_replacing_ckpt_with_a_symlink_to_an_outside_dir_is_not_read(tmp_path):
    """The base directory is job-controlled too: ckpt/ itself may be swapped for a link after the manifest was read."""
    out = tmp_path / "outside"; out.mkdir()
    (out / "MANIFEST.json").write_text(json.dumps([dict(ckpt_id="c", step=1, shards=[dict(file="s.pt")])]))
    (out / "s.pt").write_bytes(b"outside")
    jd = tmp_path / "job"; jd.mkdir()
    os.symlink(out, jd / "ckpt")
    c = Capturer(jd, "run0"); c.scan()
    assert c.records == []
    assert not (trusted_dir(jd) / "captures").exists()


def test_the_base_directory_itself_is_never_followed(tmp_path):
    """read_nofollow(base, rel): `base` is pinned with O_NOFOLLOW too (previously resolve()d, i.e. job-controlled)."""
    out = _victim(tmp_path)
    jd = tmp_path / "job"; jd.mkdir()
    os.symlink(out, jd / "ckpt")
    with pytest.raises(OSError):
        trusted.read_nofollow(jd / "ckpt", "secret")


def test_manifest_validation_does_not_follow_a_symlinked_shard(tmp_path):
    outside = tmp_path / "x.pt"; torch.save({"model.fc1.weight": torch.zeros(1)}, outside)
    jd = tmp_path / "job"; (jd / "ckpt").mkdir(parents=True)
    os.symlink(outside, jd / "ckpt" / "s.pt")
    e = dict(ckpt_id="c", step=1, shards=[dict(file="s.pt", sha256="0")])
    with pytest.raises(manifest.ManifestError):
        manifest.check_entry(jd, e)


def test_read_file_never_returns_what_a_planted_symlink_points_at(tmp_path):
    from traindoctor_env.tools import Episode
    victim = _victim(tmp_path)
    ws = tmp_path / "ws"; (ws / "data").mkdir(parents=True); (ws / "job").mkdir()
    os.symlink(victim / "secret", ws / "data" / "loot")
    (ws / "data" / "ok.idx").write_text("fine")
    os.symlink(victim, ws / "data" / "dir")
    ep = Episode(); ep.replay = SimpleNamespace(jd=ws)
    assert "S3CRET" not in ep._read_file("data/loot") and "S3CRET" not in ep._read_file("data/dir/secret")
    listing = ep._list_files("data")
    assert "ok.idx" in listing and "loot" not in listing and "dir" not in listing
    assert "fine" in ep._read_file("data/ok.idx")


def _fake_snapshot(tmp_path):
    fx = tmp_path / "fx"
    for rel in (*EDITABLE, "job/config.py"):
        (fx / rel).parent.mkdir(parents=True, exist_ok=True)
        (fx / rel).write_text(f"pristine {rel}\n")
    (fx / "data" / "part-0").mkdir(parents=True)
    return SimpleNamespace(meta=dict(task="B", kill=[0, 5], frontier=0), fixture_dir=fx)


def test_editable_files_are_copied_without_following_symlinks(tmp_path):
    victim = _victim(tmp_path)
    snap = _fake_snapshot(tmp_path)
    ws = tmp_path / "ws"; (ws / "job").mkdir(parents=True)
    os.symlink(victim / "secret", ws / "job" / "train.py")
    (ws / "job" / "data.py").write_text("agent edit\n")
    os.symlink(victim / "secret", ws / "job" / "config.py")             # a non-editable file replaced by a link
    (tmp_path / "t").mkdir()
    jd, _, _, _, tampered = _fresh_protocol_dir(snap, ws, tmp_path / "t")
    assert "S3CRET" not in (jd / "job" / "train.py").read_text()        # pristine kept
    assert (jd / "job" / "data.py").read_text() == "agent edit\n"
    assert "job/train.py" in tampered and "job/config.py" in tampered


def test_workspace_writes_do_not_follow_symlinks(tmp_path):
    victim = _victim(tmp_path)
    rp = Replay.__new__(Replay); rp.jd = tmp_path / "ws"; rp.notes = []
    (rp.jd / "job").mkdir(parents=True)
    os.symlink(victim / "secret", rp.jd / "job" / "train.py")
    assert rp.write_file("job/train.py", "overwritten") is False
    assert (victim / "secret").read_text() == "S3CRET"


def test_lock_that_is_a_symlink_is_stale_and_removal_does_not_touch_its_target(tmp_path):
    victim = _victim(tmp_path)
    os.symlink(victim / "secret", tmp_path / ".lock")
    assert rs.clean_stale_state(tmp_path) and not os.path.lexists(tmp_path / ".lock")
    assert (victim / "secret").read_text() == "S3CRET"


# ---- finding 3: verification cannot be satisfied by replaying earlier runs --------------------------------------
SPEC = DataSpec(world=1, num_samples=8, batch_size=2, seed=5)     # 4 steps per epoch


def _values(sched, epoch, index):
    r = DataServer(SPEC, sched).serve(0, dict(op="batch", epoch=epoch, index=index))
    import base64
    return base64.b64decode(r["x"])


def test_data_values_from_the_boundary_on_differ_per_verification_but_earlier_positions_are_stable():
    from jobs.dataserver import SaltSchedule
    base = salt_for("rule")
    a, b = SaltSchedule.fresh_for(base, boundary=2), SaltSchedule.fresh_for(base, boundary=2)
    plain = lambda e, i: _values(base, e, i)   # what any earlier (exploratory) run was served  # noqa: E731
    for e, i in ((0, 0), (0, 1)):                                    # positions 0, 1: below the boundary
        assert _values(a, e, i) == plain(e, i) == _values(b, e, i)
    for e, i in ((0, 2), (0, 3), (1, 0)):                            # positions 2, 3, 4: keyed by the nonce
        assert _values(a, e, i) != plain(e, i)
        assert _values(a, e, i) != _values(b, e, i)


def _ref_and_job(tmp_path, digests, served):
    """A one-rank, two-step job whose launcher-captured checkpoints equal the reference's states exactly."""
    from traindoctor_env.reexec import state_digest_of
    torch.manual_seed(0)
    h, cfg = 2, dict(lr=0.1, hidden=2, seed=1, world_size=1, batch_size=1, ckpt_every=1, num_samples=2)
    shapes = {"model.fc1.weight": (h, 8), "model.fc1.bias": (h,), "model.fc2.weight": (4, h), "model.fc2.bias": (4,)}
    states = {}
    for step in (1, 2):
        st = {k: torch.randn(*v) for k, v in shapes.items()}
        st.update({f"opt.{i}": torch.randn(*v) for i, v in enumerate(shapes.values())})
        st.update(sampler=torch.tensor(list(divmod(step, 2))), step=torch.tensor([step]))
        states[step] = st
    ref = dict(steps=2, config=cfg, states={str(s): {k: v.tolist() for k, v in st.items()} for s, st in states.items()},
               samples={"0": {"0": [0], "1": [1]}})
    jd = tmp_path / "job"; jd.mkdir()
    caps = []
    for step, st in states.items():
        f = trusted_dir(jd) / "captures" / "run0" / f"{step:06d}.pt"
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(st, f)
        caps.append(dict(ckpt_id=f"ck{step}", step=step, sha256="x", file=str(f.relative_to(trusted_dir(jd)))))
    dig = state_digest_of(states[2])
    trusted.append_run(jd, dict(run_id="run0", start_step=0, resume_from=None, status="completed", completed=True, world=1,
                                kill_planned=False, kill_done=False, sandboxed=True, served=served, acked={"0": 2},
                                digests={"0": digests(dig)}, captures=caps))
    return jd, SimpleNamespace(reference=ref, salt=b"s"), dig


def test_state_equality_with_the_reference_is_not_enough_without_sample_coverage_and_transitions(tmp_path):
    """Rule: a run that was served NO data (or too little to have produced its checkpoints) did not train, however
    exactly its checkpoints match the reference. The fast path is held to the same coverage / transition checks."""
    jd, snap, dig = _ref_and_job(tmp_path, lambda d: d, served={"0": []})
    out = check(jd, snap, lost=0)
    assert out.states_match_reference                      # the replayed checkpoints are "perfect"...
    assert not out.samples_match_reference and not out.transitions_consistent
    assert not out.ok and out.path == ""                   # ...and still do not verify


def test_rank_agreement_is_judged_against_the_launcher_captured_state_not_reported_digests(tmp_path):
    jd, snap, dig = _ref_and_job(tmp_path, lambda d: "forged-identical", served={"0": []})
    assert check(jd, snap, lost=0).ranks_consistent is False
    (tmp_path / "again").mkdir()
    jd2, snap2, dig2 = _ref_and_job(tmp_path / "again", lambda d: d, served={"0": []})
    assert check(jd2, snap2, lost=0).ranks_consistent is True


def test_trusted_digest_has_the_format_the_ranks_report():
    import sys

    from traindoctor_env.reexec import state_digest_of
    sys.path.insert(0, str(sandbox.RUNTIME))
    try:
        import harness
    except Exception as e:  # the runtime module needs the job's environment  # noqa: BLE001
        pytest.skip(f"harness not importable here: {type(e).__name__}")
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(8, 2), torch.nn.Linear(2, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.randn(3, 8)).sum().backward(); opt.step()
    sd = {f"model.fc{i + 1}.{n}": getattr(model[i], n).detach().clone() for i in range(2) for n in ("weight", "bias")}
    sd.update({f"opt.{i}": opt.state[p]["momentum_buffer"].clone() for i, p in enumerate(model.parameters())})
    assert state_digest_of(sd) == harness.state_digest(opt)


# ---- finding 4: citations are checked against the incident snapshot, never against rewritable run0 files --------
def _rec(i, kind, rank, step, text):
    return dict(id=i, kind=kind, rank=rank, step=step, text=text, source="executed")


INCIDENT = [_rec("E000001", "log", 1, 5, "[run0] rank 1 INFO waiting for shard 1"),
            _rec("E000002", "trace", 0, 5, "[run0] rank 0 step=5 seq=9 all_reduce bucket=0 numel=4 pending"),
            _rec("E000003", "log", 0, 1, "[run0] rank 0 INFO started")]
TRUTH = Truth(1, "A2", 0, "RANK_STALLED_BEFORE_COLLECTIVE", "A", [], dict(rank=1, step=5))


def test_citing_ids_of_run0_records_the_job_rewrote_after_relaunch_fails():
    rewritten = [_rec("F000001", "log", 1, 5, "[run0] rank 1 INFO waiting for shard 99"),      # content changed by the job
                 _rec("F000002", "trace", 0, 5, "[run0] rank 0 step=5 seq=9 all_reduce bucket=0 numel=5 pending"),
                 _rec("F000003", "log", 0, 1, "[run0] rank 0 INFO started")]
    cited = ["F000001", "F000002"]
    assert not verifiers.evidence_ok([INCIDENT, rewritten], TRUTH, cited, set(cited))


def test_citing_the_unchanged_incident_records_under_their_later_phase_ids_still_counts():
    relabeled = [dict(r, id="F" + r["id"][1:]) for r in INCIDENT]
    cited = ["F000001", "F000002"]
    assert verifiers.evidence_ok([INCIDENT, relabeled], TRUTH, cited, set(cited))
