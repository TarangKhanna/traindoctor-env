"""OS isolation of job processes (macOS sandbox-exec). The profile and the fail-closed rule are checked anywhere; the
behavioural probes need a sandbox that can be applied and are skipped, with the reason, where it cannot
(Linux CI, or inside another sandbox)."""
import subprocess
import sys
import textwrap

import pytest

from jobs import sandbox
from jobs.sandbox import SandboxUnavailable, profile, wrap

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="sandbox-exec cannot be applied in this environment")


def test_profile_is_deny_by_default_and_allows_nothing_launch_related(tmp_path):
    p = profile(tmp_path)
    assert p.startswith("(version 1)\n(deny default)") and "(allow default)" not in p
    low = p.lower()
    for forbidden in ("launchservices", "lsd", "appleevent", "com.apple.coreservices", "/usr/bin/open", "osascript",
                      "process-exec*", "network-outbound)", '(allow network* )'):
        assert forbidden not in low, forbidden
    assert p.count("(allow process-exec ") == 1 and "/usr/bin/" not in p.split("(allow process-exec ")[1].split("\n")[0]
    assert f'(allow file-write* (subpath "{tmp_path.resolve()}")' in p and "(deny file-read* file-write* (regex" in p


def test_unavailable_sandbox_fails_closed_unless_explicitly_opted_out(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(SandboxUnavailable):
        wrap(["python", "x.py"], tmp_path)
    assert wrap(["python", "x.py"], tmp_path, allow_unsandboxed=True) == ["python", "x.py"]
    monkeypatch.setattr(sandbox, "available", lambda: True)
    argv = wrap(["python", "x.py"], tmp_path)
    assert argv[0] == "/usr/bin/sandbox-exec" and argv[-2:] == ["python", "x.py"]


def sandboxed(tmp_path, code):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (tmp_path / "_private").mkdir(exist_ok=True)
    (tmp_path / "_private" / "secret.txt").write_text("s3cret")
    (tmp_path / "sibling.txt").write_text("sibling")
    (run / "probe.py").write_text(textwrap.dedent(code))
    argv = wrap([sys.executable, str(run / "probe.py")], run)
    r = subprocess.run(argv, cwd=run, capture_output=True, text=True, timeout=60)
    return r.stdout.strip(), r


@needs_sandbox
def test_job_can_work_inside_its_run_dir_and_on_loopback(tmp_path):
    out, r = sandboxed(tmp_path, f"""
        import socket
        open("ok.txt", "w").write("x")
        a = socket.socket(); a.bind(("127.0.0.1", 0)); a.listen(); b = socket.socket(); b.connect(a.getsockname())
        print("works")
        """)
    assert out == "works", r.stderr


@needs_sandbox
@pytest.mark.parametrize("probe", [
    "open(%r).read()" % "../_private/secret.txt",
    "open(%r).read()" % "../sibling.txt",
    "open('../escape.txt', 'w').write('x')",
    "__import__('socket').create_connection(('93.184.216.34', 80), timeout=3)",
    "__import__('subprocess').run(['/usr/bin/open', '-a', 'Calculator'], check=True)",
    "__import__('subprocess').run(['/usr/bin/osascript', '-e', '1'], check=True)",
    "__import__('subprocess').run(['/bin/ls', '/'], check=True)",
])
def test_forbidden_actions_fail_under_the_sandbox(tmp_path, probe):
    """Reading _private/ or a sibling of the run dir, writing outside it, a non-loopback connection, launching an app,
    and executing anything but the interpreter all fail."""
    out, r = sandboxed(tmp_path, f"""
        try:
            {probe}
            print("ALLOWED")
        except BaseException as e:
            print("DENIED", type(e).__name__)
        """)
    assert out.startswith("DENIED"), (out, r.stderr)
    assert not (tmp_path / "escape.txt").exists()
