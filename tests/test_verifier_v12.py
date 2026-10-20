"""Verifier v1.2 (default): v1.1 plus history_intact. Rule: a checkpoint a run lists below the step it resumed from is
history; it must be byte-identical (id, step, launcher hash) to what the launcher captured in an earlier run. A run that
fabricates or rewrites an old checkpoint is rejected, even though the state checks never look at it (v1 and v1.1 skip
captures below a run's start step, so they accepted such a run: reproduced on 3 executed Task B incidents, local
unsandboxed exploratory run, 2026-10-20). v1 and v1.1 behaviour is unchanged."""
import pytest

from jobs.build import Edit
from traindoctor_env import reexec
from traindoctor_env.reexec import FORMAT_CHECKS, Replay, history_intact


def cap(cid, step, sha):
    return dict(ckpt_id=cid, step=step, sha256=sha, file=f"captures/x/{sha}.pt")


def ledger(second_run_old_sha="a4"):
    return [dict(run_id="run1", start_step=0, captures=[cap("ck0004", 4, "a4"), cap("ck0008", 8, "a8")]),
            dict(run_id="run2", start_step=8, captures=[cap("ck0004", 4, second_run_old_sha), cap("ck0008", 8, "a8"),
                                                        cap("ck0012", 12, "b12")])]


def test_unchanged_older_checkpoint_is_history():
    led = ledger()
    assert history_intact(led, led) and history_intact(led[1:], led)


def test_rewritten_older_checkpoint_fails():
    led = ledger(second_run_old_sha="forged")
    assert not history_intact(led[1:], led)


def test_older_checkpoint_never_captured_before_fails():
    led = ledger()
    led[1]["captures"].append(cap("ck0001", 1, "fake"))
    assert not history_intact(led[1:], led)


def test_same_hash_under_another_id_or_step_fails():
    led = ledger()
    led[1]["captures"][0] = cap("ck0004", 3, "a4")
    assert not history_intact(led[1:], led)


def test_a_capture_made_only_by_a_later_run_does_not_count():
    led = ledger()
    led.insert(0, dict(run_id="run0", start_step=8, captures=[cap("ck0004", 4, "zz")]))
    led.append(dict(run_id="run3", start_step=0, captures=[cap("ck0004", 4, "zz")]))
    assert not history_intact(led[:1], led)


def test_v12_is_registered_and_is_the_default():
    assert FORMAT_CHECKS == ("v1", "v1.1", "v1.2")
    assert reexec.check.__defaults__[-1] == "v1.2" and Replay.__init__.__defaults__[-1] == "v1.2"


def test_unknown_format_check_in_outcomes_is_refused(tmp_path):
    rp = Replay.__new__(Replay)
    rp.format_check = "v1"
    with pytest.raises(ValueError):
        rp.outcomes(("v9",))
    with pytest.raises(ValueError):
        rp.outcomes(())


FABRICATE = Edit("job/ckpt.py", "        commit(cid, step, world_size)\n", """        commit(cid, step, world_size)
        if step == 12:
            fake = {k: (torch.randn_like(v) if v.is_floating_point() else v) for k, v in state.items()}
            fake.update(sampler=torch.tensor([0, 1]), step=torch.tensor([1]))
            Path("ckpt", "ck0001").mkdir(exist_ok=True)
            torch.save(fake, Path("ckpt", "ck0001", "state.pt"))
            es = json.loads(MANIFEST.read_text())
            es.insert(0, dict(ckpt_id="ck0001", step=1, world_size=world_size, shards=[dict(file="ck0001/state.pt",
                      sha256=digest(fake), shape=list(fake["model.fc1.weight"].shape))]))
            MANIFEST.write_text(json.dumps(es))
""")


@pytest.mark.executed
@pytest.mark.parametrize("name", ["B_DUP"])
def test_gold_plus_fabricated_old_checkpoint_fails_only_v12(snapshot, needs_sandbox, name):
    from traindoctor_env.snapshot import load_truth
    from traindoctor_env.verifiers import edits_of
    snap = snapshot(name)
    gold = edits_of(load_truth(snap))
    plain = reexec.verify_fix_checks(snap, gold, FORMAT_CHECKS)
    forged = reexec.verify_fix_checks(snap, gold + [FABRICATE], FORMAT_CHECKS)
    assert all(o.ok for o in plain.values())
    assert forged["v1"].ok and forged["v1.1"].ok and not forged["v1.2"].ok and not forged["v1.2"].history_intact


@pytest.mark.parametrize("stored_sampler", [True, False])
def test_historical_v1_verdicts_are_reproducible_when_selected(tmp_path, stored_sampler):
    """A valid derived sampler repair passes the default, but v1 still requires stored sampler state."""
    from types import SimpleNamespace

    import torch

    from jobs import trusted
    from jobs.dataserver import DataServer, DataSpec

    rc = dict(lr=0.1, hidden=2, seed=7, world_size=1, batch_size=1, ckpt_every=1, num_samples=2)
    server = DataServer(DataSpec(1, 2, 1, 7), b"rule")
    server.serve(0, dict(epoch=0, index=0))
    rec = dict(run_id="run0", start_step=0, resume_from=None, completed=True, world=1, captures=[],
               served={"0": server.served[0]}, acked={"0": 1})
    st = reexec._replay(None, 0, 1, rec, rc, b"rule")
    st.update(step=torch.tensor([1]), sampler=torch.tensor([0, 1]))
    ref = dict(steps=1, config=rc, states={"1": {k: v.tolist() for k, v in st.items()}}, samples={"0": {"0": server.served[0][0][2]}})
    if not stored_sampler:
        del st["sampler"]
    rec["digests"] = {"0": reexec.state_digest_of(st)}
    capture = trusted.trusted_dir(tmp_path) / "captures" / "state.pt"
    capture.parent.mkdir(parents=True)
    torch.save(st, capture)
    rec["captures"] = [dict(ckpt_id="ck1", step=1, sha256="rule", file="captures/state.pt")]
    trusted.append_run(tmp_path, rec)
    snap = SimpleNamespace(reference=ref, salt=b"rule")
    historical = reexec.check(tmp_path, snap, 0, format_check="v1")
    assert historical.ok is stored_sampler
    assert historical.format_check == "v1"
    default = reexec.check(tmp_path, snap, 0)
    assert default.ok and default.format_check == "v1.2"
    v11 = reexec.check(tmp_path, snap, 0, format_check="v1.1")
    assert v11.ok and v11.format_check == "v1.1"


@pytest.mark.parametrize("fmt", ["v1", "v1.1", "v1.2"])
def test_wrappers_forward_selected_format_check(monkeypatch, fmt):
    from types import SimpleNamespace

    from traindoctor_env import bundle, tiera

    seen = []
    out = SimpleNamespace(format_check=fmt)
    monkeypatch.setattr(bundle, "verify_fix", lambda *a, **kw: seen.append(kw["format_check"]) or out)
    candidate = SimpleNamespace(id="F1", edits=(), destructive=False)
    assert bundle.outcome_table(None, [candidate], format_check=fmt) == {"F1": out}
    monkeypatch.setattr(bundle, "verdict", lambda *a: a[-3])
    assert bundle.answer_verdict(None, SimpleNamespace(records=[]), [candidate], {}, "X", [], "F1", True,
                                 live=object(), format_check=fmt) is out
    monkeypatch.setattr(tiera, "answer_verdict", lambda *a, **kw: seen.append(kw["format_check"]) or SimpleNamespace(v0=True, v1=True, v2=True))
    completion = '{"label":"X","evidence":[],"repair":"F1","claims_fixed":true}'
    # Use the parser's public answer shape.
    monkeypatch.setattr(tiera, "parse_answer", lambda *a: SimpleNamespace(label="X", evidence=[], repair="F1", claims_fixed=True))
    assert tiera.reward(completion, None, None, [candidate], {}, format_check=fmt).v2
    assert seen == [fmt, fmt, fmt]


def test_all_entry_points_default_to_v12():
    import inspect

    from traindoctor_env import bundle, tiera

    for fn in (reexec.check, Replay, reexec.verify_fix, bundle.outcome_table):
        assert inspect.signature(fn).parameters["format_check"].default == "v1.2"
    for fn in (bundle.answer_verdict, tiera.reward):
        assert inspect.signature(fn).parameters["format_check"].default is None
    assert inspect.signature(reexec.verify_fix_checks).parameters["format_checks"].default == ("v1.2",)


def test_old_pickled_outcome_without_format_check_reads_as_v1():
    import pickle
    from dataclasses import replace

    old = replace(reexec.NO_RELAUNCH)
    del old.__dict__["format_check"]
    restored = pickle.loads(pickle.dumps(old))
    assert "format_check" not in restored.__dict__
    assert restored.format_check == "v1"


@pytest.mark.parametrize("fmt", FORMAT_CHECKS)
@pytest.mark.parametrize("run", [None, dict(completed=False), dict(completed=True, kill_done=False),
                                 dict(completed=True, kill_done=False, fault_not_injected=True)])
def test_failed_check_records_the_selected_verifier(tmp_path, monkeypatch, fmt, run):
    from types import SimpleNamespace

    led = [] if run is None else [dict(run_id="run0", **run)]
    monkeypatch.setattr(reexec.trusted, "ledger", lambda jd: led)
    snap = SimpleNamespace(reference=dict(config=dict(ckpt_every=1), steps=1), salt=b"rule")
    out = reexec.check(tmp_path, snap, 0, expect_kill=True, format_check=fmt)
    assert not out.ok and out.format_check == fmt


@pytest.mark.parametrize("entry, live", [("answer_verdict", False), ("answer_verdict", True), ("reward", False)])
@pytest.mark.parametrize("requested", [None, "v1", "v1.2"])
@pytest.mark.parametrize("produced", ["v1", "v1.2"])
def test_scoring_respects_outcome_verifier_provenance(monkeypatch, entry, requested, live, produced):
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    from traindoctor_env import bundle, tiera

    records = [dict(id=f"E00000{i}", kind="samples", rank=0, step=0, text=f"[run{i}] samples=1") for i in (1, 2)]
    truth = SimpleNamespace(label="NO_FAULT", kind="B_OK", details=dict(rank=0))
    b = SimpleNamespace(records=records)
    menu = [SimpleNamespace(id="R00001", edits=(), destructive=False)]
    out = replace(reexec.NO_RELAUNCH, ok=True, format_check=produced)
    monkeypatch.setattr(bundle, "verify_fix", lambda *a, **kw: out)
    kwargs = {} if requested is None else dict(format_check=requested)

    def score():
        if entry == "reward":
            completion = json.dumps(dict(label=truth.label, evidence=[r["id"] for r in records], repair="R00001", claims_fixed=True))
            return tiera.reward(completion, truth, b, menu, {"R00001": out}, **kwargs).v2
        return bundle.answer_verdict(truth, b, menu, {} if live else {"R00001": out}, truth.label,
                                     [r["id"] for r in records], "R00001", True,
                                     live=object() if live else None, **kwargs).v2

    if requested is not None and requested != produced:
        with pytest.raises(ValueError, match="requested format_check"):
            score()
    else:
        assert score()


def test_verify_helpers_forward_default_and_explicit_checks(monkeypatch):
    seen = []

    class StubReplay:
        def __init__(self, snapshot, allow_unsandboxed=None, backend=None, format_check="v1.2"):
            self.format_check = format_check

        def outcome(self):
            return self.format_check

        def outcomes(self, checks):
            seen.append(checks)
            return {fc: fc for fc in checks}

        def close(self):
            pass

    monkeypatch.setattr(reexec, "Replay", StubReplay)
    assert reexec.verify_fix(None, []) == "v1.2"
    assert reexec.verify_fix(None, [], format_check="v1") == "v1"
    assert reexec.verify_fix_checks(None, []) == {"v1.2": "v1.2"}
    assert reexec.verify_fix_checks(None, [], ("v1", "v1.1")) == {"v1": "v1", "v1.1": "v1.1"}
    assert seen == [("v1.2",), ("v1", "v1.1")]


def test_default_rejects_fabricated_history_that_explicit_v1_accepts(tmp_path):
    """Two real pristine updates in synthetic launcher records; an added older entry changes only history."""
    from types import SimpleNamespace

    import torch

    from jobs import trusted
    from jobs.dataserver import DataServer, DataSpec

    rc = dict(lr=0.1, hidden=2, seed=7, world_size=1, batch_size=1, ckpt_every=1, num_samples=2)
    server = DataServer(DataSpec(1, 2, 1, 7), b"rule")
    for index in range(2):
        server.serve(0, dict(epoch=0, index=index))
        server.note_step(0, dict(op="step", n=index + 1))
    full = dict(world=1, start_step=0, served={"0": server.served[0]})
    states = {}
    for step in (1, 2):
        st = reexec._replay(None, 0, step, full, rc, b"rule")
        st.update(step=torch.tensor([step]), sampler=torch.tensor(list(divmod(step, 2))))
        states[step] = st
    ref = dict(steps=2, config=rc, states={str(s): {k: v.tolist() for k, v in st.items()} for s, st in states.items()},
               samples={"0": {str(i): served[2] for i, served in enumerate(server.served[0])}})
    old = {k: v.clone() for k, v in states[1].items()}
    old.update(step=torch.tensor([0]), sampler=torch.tensor([0, 0]))
    captures = {}
    for step, st in ((0, old), *states.items()):
        path = trusted.trusted_dir(tmp_path) / "captures" / f"{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(st, path)
        captures[step] = dict(ckpt_id=f"ck{step}", step=step, sha256=str(step), file=f"captures/{step}.pt")
    first = dict(run_id="run0", world=1, start_step=0, resume_from=None, completed=False,
                 served={"0": server.served[0][:1]}, captures=[captures[1]])
    last = dict(run_id="run1", world=1, start_step=1, resume_from="ck1", resume_sha256="1", completed=True,
                served={"0": [[*server.served[0][1][:3], 0]]}, captures=[captures[1], captures[2]],
                digests={"0": reexec.state_digest_of(states[2])}, acked={"0": 1})
    trusted.append_run(tmp_path, first)
    trusted.append_run(tmp_path, last)
    snap = SimpleNamespace(reference=ref, salt=b"rule")
    assert reexec.check(tmp_path, snap, 0).ok
    # Re-commit an entry older than the resume point that no earlier launcher ever captured.
    last["captures"].insert(0, captures[0])
    (trusted.trusted_dir(tmp_path) / "ledger.jsonl").unlink()
    trusted.append_run(tmp_path, first)
    trusted.append_run(tmp_path, last)
    assert reexec.check(tmp_path, snap, 0, format_check="v1").ok
    assert reexec.check(tmp_path, snap, 0, format_check="v1.1").ok
    default = reexec.check(tmp_path, snap, 0)
    assert not default.ok and not default.history_intact
