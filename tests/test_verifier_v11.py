"""Verifier v1.1 (opt-in `format_check='v1.1'`): the sampler position is judged by BEHAVIOUR, not by representation.
Rule: the position each rank requested right after a restore,
as recorded by the harness-owned data server, must equal the reference position for the committed step of the
checkpoint resumed from, then advance one reference step per acknowledged step; restored state, sample accounting
and every other invariant stay. Nothing a job wrote (logs, progress files, a stored position it chose) can
substitute for that. The default stays v1 (frozen): a checkpoint must store the position."""
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from jobs import trusted
from traindoctor_env import reexec
from traindoctor_env.oracle import reference_solution
from traindoctor_env.snapshot import load_truth

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("redteam_v11", ROOT / "scripts" / "redteam_suite.py")
rt = importlib.util.module_from_spec(_spec)
sys.modules["redteam_v11"] = rt
_spec.loader.exec_module(rt)

RC = dict(num_samples=96, world_size=2, batch_size=4)       # spe = 12
SPE = 96 // 2 // 4
STEPS = 12


# --- pure behaviour check on synthetic harness records ------------------------------------------------------------
def _served(base, n, world_rank_shift=0):
    return [[*divmod(base + k, SPE), [k], k] for k in range(n)]


def _fixture(base=4, steps=STEPS, state_step=None):
    cap = dict(ckpt_id="ckpt-%d" % base, step=base, sha256="aa", file="c.pt")
    first = dict(run_id="run1", start_step=0, resume_from=None, completed=False, world=2, captures=[cap],
                 served={"0": _served(0, 6), "1": _served(0, 6)})
    run2 = dict(run_id="run2", start_step=base, resume_from="ckpt-%d" % base, resume_sha256="aa", completed=True, world=2,
                captures=[], served={"0": _served(base, steps - base), "1": _served(base, steps - base)})
    state = {"step": torch.tensor([base if state_step is None else state_step])}
    return [first, run2], (lambda c: state)


def ok(proto_ledger):
    proto, load = proto_ledger
    return reexec.sampler_behaviour_ok(proto, proto, load, RC, STEPS)


def test_correct_derived_position_passes():
    assert ok(_fixture())


def test_position_off_by_one_fails():
    led, load = _fixture()
    led[1]["served"]["1"] = _served(5, STEPS - 4)
    assert not ok((led, load))


def test_position_ignoring_the_committed_step_restarts_at_zero_fails():
    led, load = _fixture()
    led[1]["served"] = {r: _served(0, STEPS - 4) for r in "01"}
    assert not ok((led, load))


def test_position_conflicting_with_the_step_one_epoch_off_fails():
    led, load = _fixture()
    led[1]["served"]["0"] = [[e + 1, i, ids, w] for e, i, ids, w in led[1]["served"]["0"]]
    assert not ok((led, load))


def test_checkpoint_step_differs_from_the_pinned_start_step_fails():
    assert not ok(_fixture(state_step=5))       # the captured checkpoint is not the committed step the run claims


def test_post_restore_training_not_continuing_fails():
    led, load = _fixture()
    led[1]["served"] = {r: _served(4, 3) for r in "01"}       # completed run that stopped consuming after 3 steps
    assert not ok((led, load))
    led, load = _fixture()
    led[1]["served"] = {r: [e for e in _served(4, 8) if e[3] != 3] for r in "01"}      # a window skipped (gap)
    assert not ok((led, load))
    led, load = _fixture()
    led[1]["served"] = {r: [[*divmod(4, SPE), [0], k] for k in range(8)] for r in "01"}       # same batch every step
    assert not ok((led, load))


def test_ranks_inconsistent_with_checkpoint_metadata_fails():
    led, load = _fixture()
    del led[1]["served"]["1"]                                  # one rank unaccounted for
    assert not ok((led, load))
    led, load = _fixture()
    led[1]["served"]["1"] = _served(6, STEPS - 4)              # ranks disagree on the position
    assert not ok((led, load))
    led, load = _fixture()
    led[1]["world"] = 1
    assert not ok((led, load))


def test_job_written_log_cannot_substitute_for_the_data_servers_record():
    """A restore that reads its position from a job-written log: whatever the log says, only what the harness saw the
    job request counts. A correct-looking log with a wrong served position fails; a wrong log with a right one passes."""
    led, load = _fixture()
    led[1]["served"]["0"] = _served(2, STEPS - 4)
    led[1]["job_progress_log"] = {"0": [0, 4], "1": [0, 4]}
    assert not ok((led, load))
    led, load = _fixture()
    led[1]["job_progress_log"] = {"0": [9, 9], "1": [9, 9]}
    assert ok((led, load))


def test_unpinned_or_missing_restore_checkpoint_fails():
    led, _ = _fixture()
    assert not reexec.sampler_behaviour_ok(led, led, lambda c: None, RC, STEPS)
    led[1]["resume_sha256"] = "bb"
    assert not reexec.sampler_behaviour_ok(led, led, lambda c: {"step": torch.tensor([4])}, RC, STEPS)


# --- stored-representation check ----------------------------------------------------------------------------------
def _state(step, sampler=True):
    st = {"model.fc1.weight": torch.ones(2), "opt.0": torch.zeros(2), "step": torch.tensor([step])}
    if sampler:
        st["sampler"] = torch.tensor(list(divmod(step, SPE)))
    return st


REF = dict(states={"1": {"model.fc1.weight": [1.0, 1.0], "opt.0": [0.0, 0.0], "sampler": [0, 1], "step": [1]}})


def test_derived_position_representation_passes_v11_fails_v1():
    st, c = _state(4, sampler=False), dict(step=4)
    assert reexec._format_ok_v11(st, c, REF, RC) and not reexec._format_ok(st, c, REF, RC)


def test_stored_correct_position_passes_both():
    st, c = _state(4), dict(step=4)
    assert reexec._format_ok_v11(st, c, REF, RC) and reexec._format_ok(st, c, REF, RC)


@pytest.mark.parametrize("bad", [torch.tensor([0, 5]), torch.tensor([1, 0]), torch.tensor([0]), torch.tensor([7, 7, 7])])
def test_stored_position_conflicting_with_step_or_fabricated_fails_v11(bad):
    st = _state(4)
    st["sampler"] = bad
    assert not reexec._format_ok_v11(st, dict(step=4), REF, RC)


def test_step_counter_mismatch_or_nonfinite_or_missing_model_fails_v11():
    assert not reexec._format_ok_v11(_state(5, sampler=False), dict(step=4), REF, RC)
    st = _state(4, sampler=False)
    st["model.fc1.weight"] = torch.tensor([float("nan"), 1.0])
    assert not reexec._format_ok_v11(st, dict(step=4), REF, RC)
    st = _state(4, sampler=False)
    del st["opt.0"]
    assert not reexec._format_ok_v11(st, dict(step=4), REF, RC)
    assert not reexec._format_ok_v11(None, dict(step=4), REF, RC)


def test_unknown_format_check_is_refused():
    with pytest.raises(ValueError):
        reexec.Replay.__init__(object.__new__(reexec.Replay), None, format_check="v2")
    assert reexec.FORMAT_CHECKS == ("v1", "v1.1")


# --- end to end on a really executed Task B incident (local Gloo, repo-owned code) ---------------------------------
@pytest.fixture(scope="module")
def runs(snapshot, tmp_path_factory):
    """name -> (snapshot, saved protocol dir, check kwargs) for gold and alt2 repairs of B_DUP, captured from a real
    re-execution; the saved dir is then judged under each format_check and mutated copies of it."""
    snap = snapshot("B_DUP")
    truth = load_truth(snap)
    _, _, gold = reference_solution(snap, truth)
    cases = {c.id: c for c in rt.build_cases(truth.kind, truth.variant, snap.files, truth.details, gold)}
    out = {}
    real = reexec.check
    for cid in ("gold", "alt2"):
        saved = {}

        def spy(jd, snap_, lost, *a, **k):
            dst = tmp_path_factory.mktemp(f"proto_{cid}") / "p"
            shutil.copytree(jd, dst)
            shutil.copytree(trusted.trusted_dir(jd), trusted.trusted_dir(dst))
            saved.update(jd=dst, args=(snap_, lost, *a), kw=k)
            return real(jd, snap_, lost, *a, **k)

        reexec.check = spy
        try:
            rp = reexec.Replay(snap, allow_unsandboxed=True)
            for e in cases[cid].edits:
                rp.edit(e)
            rp.outcome()
            rp.close()
        finally:
            reexec.check = real
        out[cid] = saved
    return out


def judge(saved, fmt, jd=None):
    k = {a: b for a, b in saved["kw"].items() if a != "format_check"}
    return reexec.check(jd or saved["jd"], *saved["args"], format_check=fmt, **k)


@pytest.mark.executed
def test_gold_passes_under_both_verifiers(runs):
    assert judge(runs["gold"], "v1").ok and judge(runs["gold"], "v1.1").ok


@pytest.mark.executed
def test_alt2_derived_position_passes_v11_and_fails_v1(runs):
    o1, o11 = judge(runs["alt2"], "v1"), judge(runs["alt2"], "v1.1")
    assert not o1.ok and o1.samples_match_reference and o1.transitions_consistent and o1.restores_consistent
    assert o11.ok and o11.sampler_behaviour and o11.path == "invariants"


def _mutated(runs, tmp_path, cid="alt2"):
    dst = tmp_path / "m"
    shutil.copytree(runs[cid]["jd"], dst)
    shutil.copytree(trusted.trusted_dir(runs[cid]["jd"]), trusted.trusted_dir(dst))
    return dst


def _rewrite_ledger(jd, fn):
    p = trusted.trusted_dir(jd) / "ledger.jsonl"
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    fn(recs)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _resumed(recs):
    return next(r for r in recs if r["resume_from"] is not None)


@pytest.mark.executed
@pytest.mark.parametrize("mut", ["off_by_one", "ignores_step", "skip_window", "stalls", "rank_missing", "ranks_disagree"])
def test_real_ledger_position_mutants_fail_v11(runs, tmp_path, mut):
    jd = _mutated(runs, tmp_path)

    def f(recs):
        r = _resumed(recs)
        s0, s1 = r["served"]["0"], r["served"]["1"]
        if mut == "off_by_one":
            for e in s0:
                e[1] += 1
        elif mut == "ignores_step":
            for k, e in enumerate(s0):
                e[0], e[1] = 0, k
        elif mut == "skip_window":
            r["served"]["0"] = [e for e in s0 if e[3] != 1]
        elif mut == "stalls":
            r["served"]["0"] = s0[:1]
        elif mut == "rank_missing":
            r["served"]["1"] = []
        elif mut == "ranks_disagree":
            for e in s1:
                e[1] = (e[1] + 1) % 12
    _rewrite_ledger(jd, f)
    assert not judge(runs["alt2"], "v1.1", jd).ok


@pytest.mark.executed
def test_real_ledger_position_correct_but_samples_duplicated_fails_v11(runs, tmp_path):
    """The position record is right, but the ids actually served (the samples consumed) repeat: sample accounting
    still rejects it; the position check never stands in for it."""
    jd = _mutated(runs, tmp_path)

    def f(recs):
        r = _resumed(recs)
        for e in r["served"]["0"]:
            e[2] = list(r["served"]["0"][0][2])
    _rewrite_ledger(jd, f)
    o = judge(runs["alt2"], "v1.1", jd)
    assert o.sampler_behaviour and not o.samples_match_reference and not o.ok


@pytest.mark.executed
def test_real_checkpoint_fabricated_or_conflicting_stored_position_fails_v11(runs, tmp_path):
    for variant in ("conflict", "fabricated"):
        jd = _mutated(runs, tmp_path / variant)
        led = trusted.ledger(jd)
        for r in led:
            for c in r["captures"]:
                p = trusted.trusted_dir(jd) / c["file"]
                st = torch.load(p, weights_only=True)
                st["sampler"] = torch.tensor([0, (c["step"] + 1) % 12]) if variant == "conflict" else torch.tensor([3, 3, 3, 3])
                torch.save(st, p)
        assert not judge(runs["alt2"], "v1.1", jd).ok, variant


@pytest.mark.executed
def test_real_checkpoint_metadata_inconsistent_with_ranks_fails_v11(runs, tmp_path):
    jd = _mutated(runs, tmp_path)
    _rewrite_ledger(jd, lambda recs: _resumed(recs).update(world=1))
    assert not judge(runs["alt2"], "v1.1", jd).ok


@pytest.mark.executed
def test_real_checkpoint_step_counter_inconsistent_with_capture_fails_v11(runs, tmp_path):
    jd = _mutated(runs, tmp_path)
    c = trusted.ledger(jd)[0]["captures"][0]
    p = trusted.trusted_dir(jd) / c["file"]
    st = torch.load(p, weights_only=True)
    st["step"] = st["step"] + 1
    torch.save(st, p)
    assert not judge(runs["alt2"], "v1.1", jd).ok


@pytest.mark.executed
def test_default_and_frozen_verifiers_untouched():
    import inspect
    from traindoctor_env import verifiers
    assert inspect.signature(reexec.check).parameters["format_check"].default == "v1"
    assert inspect.signature(reexec.Replay.__init__).parameters["format_check"].default == "v1"
    assert inspect.signature(reexec.verify_fix).parameters["format_check"].default == "v1"
    assert "format_check" not in Path(verifiers.__file__).read_text() if hasattr(verifiers, "__file__") else True
