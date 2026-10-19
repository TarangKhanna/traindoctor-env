"""Rule: the status-only baselines are functions of bundle text alone (a str), and deterministic. They must not
accept scenario/truth objects. Fixtures are real bundles saved under tests/fixtures/."""
from pathlib import Path

import pytest

from audit.status_only_baseline import classify_job, classify_process

FIX = Path(__file__).parent / "fixtures"
TEXTS = {n: (FIX / f"status_bundle_{n}.txt").read_text() for n in ("a1_src", "a1_cfg", "a2")}


@pytest.mark.parametrize("fn", [classify_process, classify_job])
def test_rejects_non_text(fn):
    class Truth:
        kind = "A2"
    for bad in (Truth(), {"kind": "A2"}, 80902, None):
        with pytest.raises(TypeError):
            fn(bad)


@pytest.mark.parametrize("fn", [classify_process, classify_job])
def test_deterministic(fn):
    for t in TEXTS.values():
        assert len({fn(t) for _ in range(5)}) == 1


def test_rule_on_fixtures():
    # rule (audit/status_only_rule.md): a watchdog-killed rank marks A2; all-failed ranks default to A1_src
    assert classify_process(TEXTS["a2"]) == "A2"
    assert classify_process(TEXTS["a1_src"]) == classify_process(TEXTS["a1_cfg"]) == "A1_src"
    assert classify_job(TEXTS["a2"]) == classify_job(TEXTS["a1_src"]) == classify_job(TEXTS["a1_cfg"]) == "A1_src"
