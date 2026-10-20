"""Rule: malformed agent output (an action or arguments of the wrong shape) gets an error observation and leaves the
recorded state unchanged: nothing recorded or executed; each attempt still consumes the call budget."""
from types import SimpleNamespace

import pytest

from traindoctor_env.tools import Episode

RECORD = dict(id="E000001", kind="job", rank=None, step=None, text="job failed")

MALFORMED = [
    ["job_status"],                                   # action is not an object
    "job_status",
    None,
    {"tool": ["job_status"]},                         # tool name is not a string (unhashable)
    {"tool": 3},
    {"tool": "job_status", "args": [1]},              # args is not an object
    {"tool": "job_status", "args": None},
    {"tool": "job_status", "args": "x"},
    {"tool": "write_file", "args": {"path": "job/train.py"}},           # missing required argument
    {"tool": "collective_trace", "args": {"step": True}},               # bool is not an int
    {"tool": "rank_timeline", "args": {"window": [1]}},
    {"tool": "submit_diagnosis", "args": {"label": "X", "evidence_ids": [1], "claimed_fixed": False,
                                          "fix_summary": ""}},
    {"tool": "finish", "args": {"now": True}},                          # unexpected argument
]


def _episode() -> Episode:
    ep = Episode()
    ep.reset(SimpleNamespace(records=[RECORD]))
    return ep


@pytest.mark.parametrize("action", MALFORMED, ids=repr)
def test_malformed_action_is_an_error_and_does_not_change_the_episode(action):
    ep = _episode()
    before = ep.state()
    obs, done = ep.step(action)
    assert obs.startswith("error:") and not done
    assert ep.state() == before and ep.replay is None


def test_a_valid_action_after_malformed_ones_still_works():
    ep = _episode()
    for a in MALFORMED:
        ep.step(a)
    obs, done = ep.step({"tool": "job_status"})
    assert "E000001 job failed" in obs and not done
    assert ep.state()["calls"] == [dict(tool="job_status", args={})]


@pytest.mark.parametrize("content", ["\ud800", "a\ud800z", "\udfff"])
@pytest.mark.parametrize("existing_replay", [False, True])
def test_unencodable_write_is_an_argument_error_without_changing_the_episode(content, existing_replay):
    ep = _episode()
    ep.replay = SimpleNamespace() if existing_replay else None
    before, replay = ep.state(), ep.replay
    obs, done = ep.step(dict(tool="write_file", args=dict(path="job/train.py", content=content)))
    assert obs.startswith("error: invalid arguments:") and not done
    assert ep.state() == before and ep.replay is replay
    assert ep.tool_calls == 1
