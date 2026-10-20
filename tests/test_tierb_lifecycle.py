"""Rules for the Tier B episode lifecycle (CPU only; the workspace is a stub, no job is executed):
infrastructure failures terminate without a repair verdict; a finished episode is inert; reset leaves nothing of the previous
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


@pytest.mark.parametrize("tool,kwargs", [("relaunch", dict(fail_relaunch=True)), ("finish", dict(fail_outcome=True))])
def test_infrastructure_failure_is_terminal_and_has_no_repair_verdict(tool, kwargs):
    ep = _episode(FakeReplay(**kwargs))
    assert ep.step({"tool": tool}) == ("infrastructure_error", True)
    assert ep.calls == ep.actions == [] and ep.phase == 0 and len(ep.phase_records) == 1
    assert ep.result["status"] == "infrastructure_error"
    assert ep.result["verified_fixed"] is None and ep.result["claimed_vs_verified"] is None
    frozen = ep.state()
    assert ep.step({"tool": "finish"}) == ("episode finished", True)
    assert ep.state() == frozen


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


def test_episode_is_bounded_in_tool_calls():
    ep = _episode(FakeReplay())
    for _ in range(5000):
        _, done = ep.step({"tool": "job_status"})
        if done:
            assert ep.result["status"] == "budget_exhausted"
            assert ep.result["verified_fixed"] is None
            assert len(ep.calls) == ep.max_tool_calls == 200
            return
    raise AssertionError("5000 tool calls accepted")


def test_relaunch_budget_prevents_another_execution(monkeypatch):
    monkeypatch.setattr("traindoctor_env.tools.public_records", lambda *args: [RECORD])
    rp = FakeReplay()
    ep = Episode(max_relaunches=2)
    ep.reset(SimpleNamespace(records=[RECORD], seed=0))
    ep.replay = rp
    for _ in range(2):
        assert ep.step({"tool": "relaunch"}) == ("recorded", False)
    assert ep.step({"tool": "relaunch"}) == ("budget_exhausted", True)
    assert rp.relaunches == 2 and ep.phase == 2 and len(ep.calls) == 2
    assert ep.result["verified_fixed"] is None


def test_invalid_calls_and_rejected_writes_still_consume_the_budget():
    ep = Episode(max_tool_calls=2, max_relaunches=0)
    ep.reset(SimpleNamespace(records=[RECORD], seed=0))
    assert not ep.step(None)[1]
    assert not ep.step({"tool": "write_file", "args": {"path": "job/a", "content": "x" * 65537}})[1]
    assert ep.step({"tool": "finish"}) == ("budget_exhausted", True)
    assert ep.replay is None and ep.calls == []
    ep.reset(SimpleNamespace(records=[RECORD], seed=0))
    assert ep.step({"tool": "relaunch"}) == ("budget_exhausted", True)


@pytest.mark.parametrize("content", ["x" * 65537, "é" * 32769])
def test_write_size_is_capped_in_bytes_before_creating_a_workspace(content):
    ep = _episode()
    obs, done = ep.step({"tool": "write_file", "args": {"path": "job/a.py", "content": content}})
    assert obs.startswith("error:") and not done
    assert ep.replay is None and ep.calls == []


def test_write_at_the_byte_limit_is_accepted():
    rp = FakeReplay()
    ep = _episode(rp)
    text = "é" * 32768
    assert ep.step({"tool": "write_file", "args": {"path": "job/a.py", "content": text}}) == ("recorded", False)
    assert rp.files == {"job/a.py": text}


def test_grep_uses_bounded_literal_patterns():
    ep = _episode()
    ep.records = [dict(RECORD, kind="log", rank=0, step=3, text="waiting for shard"),
                  dict(RECORD, id="E000002", kind="log", rank=1, step=4, text="a" * 10000 + "!")]
    assert "waiting for shard" in ep.step({"tool": "grep_logs", "args": {"pattern": "waiting", "ranks": [0], "window": [3, 3]}})[0]
    assert ep.step({"tool": "grep_logs", "args": {"pattern": "(a+)+$"}}) == ("(no matching records)", False)
    assert ep.step({"tool": "grep_logs", "args": {"pattern": "x" * 257}})[0].startswith("error: invalid pattern")


@pytest.mark.parametrize("kwargs", [dict(max_tool_calls=0), dict(max_tool_calls=True), dict(max_tool_calls=float("inf")),
                                    dict(max_relaunches=-1), dict(max_relaunches=1.5)])
def test_budget_arguments_are_validated(kwargs):
    with pytest.raises(ValueError):
        Episode(**kwargs)


def test_reset_restores_both_budgets():
    ep = Episode(max_tool_calls=1, max_relaunches=0)
    snap = SimpleNamespace(records=[RECORD], seed=0)
    ep.reset(snap)
    ep.step({"tool": "relaunch"})
    ep.reset(snap)
    assert ep.tool_calls == ep.relaunches == 0
    assert ep.step({"tool": "job_status"}) == ("E000001 job failed", False)


def test_unsandboxed_finish_is_infrastructure_error_not_wrong_repair(monkeypatch):
    rp = FakeReplay(ok=False)
    out = rp.outcome()
    out.sandboxed = False
    monkeypatch.setattr(rp, "outcome", lambda: out)
    ep = _episode(rp)
    assert ep.step({"tool": "finish"}) == ("infrastructure_error", True)
    assert ep.result["verified_fixed"] is None
