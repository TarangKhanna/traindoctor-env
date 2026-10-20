"""Rules for the Tier B episode lifecycle (CPU only; the workspace is a stub, no job is executed):
a step that raises leaves no trace of the call; a finished episode is inert; reset leaves nothing of the previous
episode; finish without a diagnosis is 'no_claim'; the final reward comes from the final workspace."""
from types import SimpleNamespace

import pytest

from traindoctor_env.reexec import FixOutcome
from traindoctor_env.tools import Episode

RECORD = dict(id="E000001", kind="job", rank=None, step=None, text="job failed")


class FakeReplay:
    def __init__(self, fail_relaunch=False, fail_outcome=False, ok=True):
        self.jd = "/nonexistent"
        self.closed = 0
        self.files, self.relaunches = {}, 0
        self.fail_relaunch, self.fail_outcome, self.ok = fail_relaunch, fail_outcome, ok

    def write_file(self, rel, text):
        self.files[rel] = text
        return True

    def relaunch(self, resume_from=None, chaos=False):
        if self.fail_relaunch:
            raise RuntimeError("sandbox unavailable")
        self.relaunches += 1

    def outcome(self):
        if self.fail_outcome:
            raise RuntimeError("sandbox unavailable")
        return FixOutcome(ok=self.ok, bit_identical=False, params_match_reference=False, samples_match_reference=False,
                          restores_consistent=False, no_duplicate_updates=False, lost_steps=0, ranks_consistent=False)

    def close(self):
        self.closed += 1


def _episode(replay=None) -> Episode:
    ep = Episode()
    ep.reset(SimpleNamespace(records=[RECORD], seed=0))
    ep.replay = replay
    return ep


def test_step_before_reset_is_an_error_not_an_attribute_error():
    obs, done = Episode().step({"tool": "job_status"})
    assert obs.startswith("error:") and not done


def test_relaunch_that_raises_leaves_no_recorded_call():
    ep = _episode(FakeReplay(fail_relaunch=True))
    before = ep.state()
    with pytest.raises(RuntimeError):
        ep.step({"tool": "relaunch", "args": {}})
    assert ep.state() == before and ep.phase == 0 and len(ep.phase_records) == 1


def test_finish_that_raises_leaves_the_episode_open_and_unrecorded():
    ep = _episode(FakeReplay(fail_outcome=True))
    before = ep.state()
    with pytest.raises(RuntimeError):
        ep.step({"tool": "finish"})
    assert ep.state() == before and not ep.done


def test_finished_episode_is_inert():
    ep = _episode(FakeReplay())
    ep.step({"tool": "finish"})
    frozen = ep.state()
    for a in ({"tool": "write_file", "args": {"path": "job/a.py", "content": "x"}}, {"tool": "relaunch"},
              {"tool": "submit_diagnosis", "args": dict(label="X", evidence_ids=[], claimed_fixed=True, fix_summary="")},
              {"tool": "finish"}, {"tool": "job_status"}):
        assert ep.step(a) == ("episode finished", True)
    assert ep.state() == frozen and ep.replay.relaunches == 0 and ep.replay.files == {}


def test_finish_without_submit_is_no_claim_and_still_verified():
    ep = _episode(FakeReplay(ok=False))
    ep.step({"tool": "finish"})
    assert ep.result["claimed_vs_verified"] == "no_claim" and ep.result["verified_fixed"] is False


def test_claim_is_compared_with_the_final_workspace_not_an_earlier_one():
    rp = FakeReplay(ok=True)
    ep = _episode(rp)
    ep.step({"tool": "submit_diagnosis", "args": dict(label="X", evidence_ids=[], claimed_fixed=True, fix_summary="")})
    rp.ok = False  # the workspace is degraded after the claim; finish must verify what is there now
    ep.step({"tool": "finish"})
    assert ep.result["claimed_vs_verified"] == "overclaim"


def test_reset_closes_the_old_workspace_and_clears_all_episode_state():
    rp = FakeReplay()
    ep = _episode(rp)
    ep.step({"tool": "job_status"})
    ep.step({"tool": "relaunch"})
    ep.step({"tool": "finish"})
    ep.reset(SimpleNamespace(records=[RECORD], seed=0))
    assert rp.closed == 1 and ep.replay is None
    assert ep.state() == dict(calls=[], actions=[], observed_ids=[], diagnosis=None, done=False, result=None)
    assert ep.phase == 0 and len(ep.phase_records) == 1


@pytest.mark.xfail(strict=True, reason="no step/relaunch budget: the episode never ends on its own (see docs/TIERB_READINESS.md)")
def test_episode_is_bounded_in_tool_calls():
    ep = _episode(FakeReplay())
    for _ in range(5000):
        _, done = ep.step({"tool": "job_status"})
        if done:
            return
    raise AssertionError("5000 tool calls accepted")
