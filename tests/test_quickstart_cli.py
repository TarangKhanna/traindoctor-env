"""Quickstart CLI contract (no Modal, no sandbox, nothing executed): the --modal path must replay the scenario and the
gold/fake pair that the local demo builds, and the modal client must not require TD_MODAL_SITE."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TD_KEY", "ab" * 8)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qs = _load("quickstart")
rt = _load("redteam_suite")


def test_defaults_route_to_local_a1_with_f11():
    a = qs.parse_args([])
    assert (a.seed, a.fake, a.modal) == (123, "F11", False)


def test_flags_are_parsed():
    a = qs.parse_args(["--seed", "7", "--fake", "F04", "--modal", "--out", "x"])
    assert (a.seed, a.fake, a.modal, a.out) == (7, "F04", True, "x")


@pytest.mark.parametrize("seed,fake", [(123, "F11"), (7, "F04"), (123, "")])
def test_modal_command_carries_the_local_seed_and_pair(seed, fake):
    cmd = qs.modal_command(seed, fake)
    assert cmd[1].endswith("scripts/redteam_suite.py")
    assert cmd[cmd.index("--seed") + 1] == str(seed)
    assert "--tasks" not in cmd, "a fixed task (the old A2 case) would diverge from the local scenario"
    assert cmd[cmd.index("--cases") + 1] == (f"gold,{fake}" if fake else "gold")


def test_modal_path_selects_the_same_scenario_as_local(capsys):
    seed = 123
    cmd = qs.modal_command(seed, "F11")
    assert rt.main(["--list", "--seed", cmd[cmd.index("--seed") + 1]]) == 0
    assert rt.SCENARIOS[f"seed{seed}"] == qs.make_scenario(seed)
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"seed{seed} ")
    ids = line.split("cases:")[1].split()
    assert "gold" in ids and "F11" in ids


def test_seed_selection_needs_no_modal_site(capsys, monkeypatch):
    monkeypatch.delenv("TD_MODAL_SITE", raising=False)
    assert rt.main(["--list", "--seed", "123"]) == 0
    assert "seed123" in capsys.readouterr().out


def test_missing_modal_client_without_override_exits_with_instructions(monkeypatch):
    monkeypatch.delenv("TD_MODAL_SITE", raising=False)
    monkeypatch.setenv("TD_KEY", "ab" * 8)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a, **k: None if n == "modal" else real(n, *a, **k))
    with pytest.raises(SystemExit) as e:
        rt.run(rt.argparse.Namespace(tasks=[], run="t", cap_usd=0.1, verifier="v1"))
    assert "pip install modal" in str(e.value)
