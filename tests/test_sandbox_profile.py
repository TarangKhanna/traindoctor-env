"""The sandbox profile may execute only the interpreter, including the framework build's second binary.

Rule: process-exec is limited to the interpreter's own binaries. A python.org framework `bin/python` re-executes
`Resources/Python.app/Contents/MacOS/Python`, so that file must be allowed, and nothing else may be.
"""
import re
import sys
from pathlib import Path

from jobs import sandbox


def _exec_rule(profile_text: str) -> str:
    return next(line for line in profile_text.splitlines() if line.startswith("(allow process-exec"))


def test_framework_python_app_binary_is_executable_and_nothing_else_is_added(tmp_path, monkeypatch):
    prefix = tmp_path / "Python.framework" / "Versions" / "3.12"
    app = prefix / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    app.parent.mkdir(parents=True)
    app.write_text("")
    monkeypatch.setattr(sys, "base_prefix", str(prefix))
    run = tmp_path / "run"
    run.mkdir()
    rule = _exec_rule(sandbox.profile(run))
    literals = re.findall(r'\(literal "([^"]+)"\)', rule)
    assert str(app.resolve()) in literals
    allowed_roots = {str(Path(sys.executable).resolve()), str(Path(sys.executable).absolute()), str(app.resolve())}
    assert set(literals) == allowed_roots          # exactly the interpreter and the framework binary


def test_without_a_framework_app_the_exec_rule_is_just_the_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
    run = tmp_path / "run"
    run.mkdir()
    literals = re.findall(r'\(literal "([^"]+)"\)', _exec_rule(sandbox.profile(run)))
    assert set(literals) == {str(Path(sys.executable).resolve()), str(Path(sys.executable).absolute())}


def test_a_framework_app_inside_the_run_directory_is_never_allowed(tmp_path, monkeypatch):
    """An agent can write inside the run directory, so a binary there must not become executable by the profile."""
    run = tmp_path / "run"
    prefix = run / "fake_prefix"
    app = prefix / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    app.parent.mkdir(parents=True)
    app.write_text("")
    monkeypatch.setattr(sys, "base_prefix", str(prefix))
    literals = re.findall(r'\(literal "([^"]+)"\)', _exec_rule(sandbox.profile(run)))
    assert str(app.resolve()) not in literals
