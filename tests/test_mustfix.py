"""Regression tests for the Codex-review must-fix items (spec M1_v2 section 6), restated for the v2 job."""
import re

import pytest

from jobs.build import apply_edits
from jobs.config import dumps, flatten, loads
from jobs.resubmit import SCRATCH
from tests.conftest import SCEN
from traindoctor_env.oracle import reference_solution
from traindoctor_env.reexec import Replay, lost_work
from traindoctor_env.scenario import job_spec, make_scenario
from traindoctor_env.snapshot import load_truth


def public_ints(sc):
    """Every integer a job's public config can show for this scenario, without executing it."""
    spec = job_spec(sc)
    text = dumps(spec.config)
    for e in spec.bug:
        if e.path == "job/config.yaml":
            text = text.replace(e.old, e.new)
    cfg, out, seen = loads(text), {}, {}
    for key, val in flatten(cfg).items():
        name = ".".join("#" if p.isdigit() else p for p in key.split("."))
        i = seen[name] = seen.get(name, -1) + 1
        out[f"{name}[{i}]"] = int(re.sub(r"\D", "", str(val)) or -1)
        out[f"{name}.key[{i}]"] = int(re.sub(r"\D", "", key.rsplit(".", 1)[-1]) or -1)
    return {**out, "scenario_seed": sc.seed, "job_seed": sc.job_seed, "world_size": sc.world_size}


def test_no_public_integer_predicts_the_label():
    """Rule: the label is independent of every public value. A residue-class predictor fit on the first half of
    the scenarios must not beat always guessing the most common label (on the second half) by more than a margin,
    for every public integer and modulus."""
    scs = [make_scenario(s) for s in range(300)]
    rows = [(public_ints(s), s.label) for s in scs]
    fit, test = rows[:150], rows[150:]
    base = max(sum(lab == c for _, lab in test) for c in {lab for _, lab in test}) / len(test)
    worst = 0.0
    for key in sorted({k for ints, _ in rows for k in ints}):
        val = lambda ints: ints.get(key, -1)  # noqa: E731
        for k in range(2, 9):
            table: dict[int, dict[str, int]] = {}
            for ints, lab in fit:
                c = table.setdefault(val(ints) % k, {})
                c[lab] = c.get(lab, 0) + 1
            guess = {r: max(c, key=c.get) for r, c in table.items()}
            acc = sum(guess.get(val(ints) % k) == lab for ints, lab in test) / len(test)
            worst = max(worst, acc - base)
            assert acc - base <= 0.15, (key, k, acc, base)
    assert len({s.label for s in scs}) == 4 and all(s.job_seed != s.seed for s in scs)


@pytest.mark.executed
def test_lost_work_counts_every_relaunch_attempt(snapshot):
    """Rule: lost work = steps recomputed beyond the last valid checkpoint over EVERY attempt. Repairing and
    relaunching from the latest checkpoint is fine, and so is repeating that completed relaunch (it recomputes nothing); but a relaunch
    from step 0 (which completes) followed by a clean one from the latest checkpoint must not launder the wasted
    attempt."""
    snap = snapshot("A2")
    _, _, edits = reference_solution(snap, load_truth(snap))
    rp = Replay(snap)
    try:
        for e in edits:
            rp.edit(e)
        rp.relaunch(None)
        rp.relaunch(None)  # resumes from the final checkpoint: nothing to recompute
        assert lost_work(rp.jd, snap.meta["frontier"], rp.runs0) <= snap.config["ckpt_every"]
        rp.relaunch(SCRATCH)
        rp.relaunch(None)
        out = rp.outcome()
        assert not out.ok and out.lost_steps > snap.config["ckpt_every"]
        assert out.bit_identical and out.samples_match_reference  # the protocol run is clean; only lost work fails it
    finally:
        rp.close()


@pytest.mark.executed
def test_task_b_counts_every_recovery_attempt_but_the_one_validation_rerun(snapshot):
    """Rule (item 3): lost work is charged over ALL recovery attempts for Task B as well. The one from-scratch rerun the
    repair legitimately needs (B_VALIDATION_SCRATCH_RUNS, a documented constant) is exempt; a second scratch rerun, or
    a third, is not: two scratch relaunches then a correct repair FAIL on lost work even though the protocol run is clean."""
    from traindoctor_env.reexec import B_VALIDATION_SCRATCH_RUNS
    snap = snapshot("B_DUP")
    _, _, edits = reference_solution(snap, load_truth(snap))

    def attempts(n):
        rp = Replay(snap)
        try:
            for e in edits:
                rp.edit(e)
            for _ in range(n):
                rp.relaunch(SCRATCH, chaos=True)
            return rp.outcome()
        finally:
            rp.close()

    one, two = attempts(B_VALIDATION_SCRATCH_RUNS), attempts(B_VALIDATION_SCRATCH_RUNS + 1)
    assert one.ok and one.lost_steps <= snap.config["ckpt_every"]
    assert not two.ok and two.lost_steps > snap.config["ckpt_every"]
    assert two.states_match_reference and two.samples_match_reference      # only lost work fails it
    assert two.relaunches > one.relaunches
