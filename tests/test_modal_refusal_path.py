"""The Modal replay path carries the launcher's refusal reason from the sandbox entry back into FixOutcome.detail.

A fake `modal` module runs the real ENTRY script, tar round-trip and host-side parsing locally (paths /work and /app
are remapped to a temp dir); only the refusal case is exercised, where no worker process is started."""
import subprocess
import sys
import types
from pathlib import Path

from traindoctor_env import modal_backend
from traindoctor_env.reexec import refusal_detail

ROOT = Path(__file__).resolve().parent.parent


class _Proc:
    def __init__(self, argv, box):
        self.argv, self.box, self._in = argv, box, []
        self.stdin = types.SimpleNamespace(write=self._in.append, write_eof=lambda: None, drain=lambda: None)

    def wait(self):
        argv = [sys.executable if a == "python" else self.box.real(a) for a in self.argv]
        r = subprocess.run(argv, input="".join(self._in), capture_output=True, text=True, timeout=120)
        self.stdout = types.SimpleNamespace(read=lambda: r.stdout)
        self.stderr = types.SimpleNamespace(read=lambda: r.stderr)
        return r.returncode


class _Box:
    created: list = []

    def __init__(self, root: Path, **kw):
        self.root, self.kw, self.object_id = root, kw, "sb-fake"
        (root / "work").mkdir(parents=True)
        (root / "app").mkdir()
        (root / "app" / "jobs").symlink_to(ROOT / "jobs")
        fs = types.SimpleNamespace()
        fs.write_bytes = lambda data, p: Path(self.real(p)).write_bytes(data)
        fs.write_text = lambda text, p: Path(self.real(p)).write_text(self.remap(text))
        fs.read_bytes = lambda p: Path(self.real(p)).read_bytes()
        self.filesystem = fs
        _Box.created.append(self)

    def remap(self, text: str) -> str:
        return text.replace('"/work', f'"{self.root}/work').replace('"/app', f'"{self.root}/app')

    def real(self, p: str) -> str:
        return str(self.root) + p if p.startswith(("/work", "/app")) else p

    def exec(self, *argv, timeout=None):
        return _Proc(list(argv), self)

    def terminate(self):
        pass


def _fake_modal(root: Path):
    m = types.ModuleType("modal")
    m.App = types.SimpleNamespace(lookup=lambda name, create_if_missing=False: object())
    m.Sandbox = types.SimpleNamespace(create=lambda **kw: _Box(root, **kw))
    return m


def test_modal_path_reports_bad_config_refused_before_any_worker(tmp_path, monkeypatch):
    """Rule: on the verified Modal path a launcher refusal is a non-completion whose detail names BAD_CONFIG and the
    field, the sandbox is created network-less with no secrets, and no worker run is recorded."""
    from jobs.build import Edit
    from traindoctor_env.reexec import verify_fix
    from traindoctor_env.scenario import make_scenario
    from traindoctor_env.snapshot import execute
    snap = execute(next(s for s in map(make_scenario, range(300)) if s.kind == "B_SKIP"), tmp_path / "s")
    monkeypatch.setitem(sys.modules, "modal", _fake_modal(tmp_path / "sb"))
    monkeypatch.setattr(modal_backend, "image", lambda: object())
    monkeypatch.setattr(modal_backend, "_IMAGE", None)
    _Box.created.clear()
    w = snap.config["world_size"]
    out = verify_fix(snap, [Edit("job/config.yaml", f"world_size: {w}", "world_size: 100000")], backend="modal")
    box = _Box.created[-1]
    assert box.kw["block_network"] is True and box.kw["secrets"] == []
    assert not out.ok and out.exec_mode == "modal_sandbox" and out.steps_completed == 0
    assert out.detail == "BAD_CONFIG: world_size must be an integer in [1, 16]"   # detail is set only when no run was recorded


def test_entry_result_parses_only_the_last_json_line():
    """Rule: a missing, malformed or non-object result line yields {} and only string fields are kept."""
    assert modal_backend.entry_result("") == {}
    assert modal_backend.entry_result("noise\nnot json") == {}
    assert modal_backend.entry_result("[1, 2]") == {}
    line = '{"ok": false, "status": "refused", "error_class": "BAD_CONFIG", "message": "x", "stderr": {}}'
    assert modal_backend.entry_result("log\n" + line) == dict(status="refused", error_class="BAD_CONFIG", message="x")


def test_refusal_text_outside_the_schema_grammar_is_never_quoted():
    """Rule: only BadConfig's fixed schema messages reach detail; any other text (lock file, sandbox output) does not."""
    ns = types.SimpleNamespace
    assert refusal_detail(ns(status="refused", error_class="BAD_CONFIG",
                             message="batch_size must be an integer in [1, 256]")) == "BAD_CONFIG: batch_size must be an integer in [1, 256]"
    assert refusal_detail(ns(status="refused", error_class="BAD_CONFIG", message="I6 worker failure")) == "BAD_CONFIG"
    assert refusal_detail(ns(status="refused", error_class="I6 worker failure", message="")) == "REFUSED"
    assert refusal_detail(ns()) == ""


def test_every_real_bad_config_message_is_quoted_and_invented_fields_are_not():
    """Rule: each message jobs.config actually raises is reported verbatim; a well-formed message about a field the
    launcher never checks is not."""
    import pytest

    from jobs.config import BadConfig, validate_config
    import importlib.util
    spec = importlib.util.spec_from_file_location("launch_limits", ROOT / "tests" / "test_launch_limits.py")
    ll = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ll)
    LIMITS, config, set_value = ll.LIMITS, ll.config, ll.set_value
    seen = 0
    for path, lower, upper in LIMITS:
        cfg = config()
        set_value(cfg, path, upper + 1)
        with pytest.raises(BadConfig) as e:
            validate_config(cfg)
        res = types.SimpleNamespace(status="refused", error_class="BAD_CONFIG", message=str(e.value))
        assert refusal_detail(res) == f"BAD_CONFIG: {e.value}"
        seen += 1
    assert seen == len(LIMITS)
    fake = types.SimpleNamespace(status="refused", error_class="BAD_CONFIG", message="invented_field must be finite in [1, 2]")
    assert refusal_detail(fake) == "BAD_CONFIG"


def test_entry_result_survives_long_output():
    """Rule: the result line is parsed from the full entry output, so a long diagnostic cannot drop it."""
    line = '{"status": "refused", "error_class": "BAD_CONFIG", "message": "m", "stderr": {"r": "' + "\\ud83d" * 3000 + '"}}'
    assert len(line) > 4000 and modal_backend.entry_result(line)["status"] == "refused"
