"""Run a job's processes under macOS sandbox-exec: agent-edited code runs on the owner's machine.

The profile is DENY BY DEFAULT and allows only:
  exec/fork     the interpreter (nothing else can be executed: no /usr/bin/open, no osascript, no shell tools)
  file reads    the run directory, the runtime package (jobs/runtime), the Python installation (stdlib +
                site-packages), and the system paths dyld and libc need; metadata (stat) anywhere, content nowhere else
  file writes   the run directory only (+ /dev/null, /dev/dtracehelper); nothing else, in particular no /tmp
  POSIX shm     only names starting with SHM_PREFIX (libomp's registration segment)
  network       loopback only (Gloo's rendezvous between ranks); the data service is an inherited socket, no port.
                Gloo's pair connections use ephemeral loopback ports the launcher cannot know in advance and
                Seatbelt filters on one port or `*`, never a range, so other loopback services are NOT closed off
                (known residual, recorded in docs/DESIGN_NOTES.md)
  mach-lookup   MACH_LOOKUPS below, nothing else (in particular no LaunchServices, no AppleEvents)
A path component named _private is unreadable even if some rule above would allow it. The reference, the metadata
and the trusted ledger are not under the run directory in the first place.

Isolation is FAIL CLOSED: asking for it where it cannot be applied raises SandboxUnavailable. Running unsandboxed is
an explicit opt-out (allow_unsandboxed=True; traindoctor_env.reexec honors TD_CI_UNSANDBOXED=1 only on Linux with CI set, and
any result produced that way is marked sandboxed=False and invalid), meant for this repository's own code and
never for agent-edited code on a developer machine.

STATUS: the profile could not be exercised while this was written (sandbox-exec cannot nest inside another sandbox);
tests/test_sandbox.py exercises it wherever the sandbox can be applied. If a denial breaks the job's Python/torch
start-up, find it with `log stream --predicate 'sender == "Sandbox"'` and add the minimal rule here."""
from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME = REPO_ROOT / "jobs" / "runtime"
MACH_LOOKUPS = (
    "com.apple.system.logger", "com.apple.logd", "com.apple.system.notification_center",
    "com.apple.system.opendirectoryd.libinfo", "com.apple.SystemConfiguration.configd",
)
SYSTEM_READS = ("/usr/lib", "/usr/share", "/System", "/private/var/db/dyld", "/private/var/db/timezone",
                "/private/etc/localtime", "/Library/Preferences/Logging")
DEVICE_READS = ("/dev/null", "/dev/urandom", "/dev/random", "/dev/dtracehelper")
# libomp (torch's OpenMP runtime) registers itself at start-up. Its binary offers three places, tried in this order
# (strings of torch/lib/libomp.dylib: "__KMP_REGISTERED_LIB_%d", "/tmp/%s", "Registering library with env var
# failed"): a POSIX shm segment named /__KMP_REGISTERED_LIB_<pid>, a file under /tmp, an environment variable. There
# is no environment switch that disables registration or moves the /tmp file (the only KMP_ variables in the binary are
# KMP_DUPLICATE_LIB_OK and KMP_LIBRARY, neither of which affects it), so the profile allows the shm name by prefix and
# NO /tmp write: if shm is refused libomp falls through to the environment variable, which needs no filesystem.
SHM_PREFIX = "/__KMP_REGISTERED_LIB_"


class SandboxUnavailable(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """Whether a sandbox can actually be applied here (it cannot on Linux, nor from inside another sandbox)."""
    if sys.platform != "darwin" or not os.access(SANDBOX_EXEC, os.X_OK):
        return False
    try:
        return subprocess.run([SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _p(p: Path | str) -> str:
    s = os.path.realpath(p)
    if '"' in s or "\\" in s:
        raise ValueError(f"unsupported path for a sandbox profile: {s!r}")
    return s


def profile(run_dir: Path) -> str:
    run = _p(run_dir)
    python = sorted({_p(sys.prefix), _p(sys.base_prefix), _p(sys.exec_prefix)})
    exe = sorted({_p(sys.executable), str(Path(sys.executable).absolute())})
    sub = lambda paths: " ".join(f'(subpath "{_p(p)}")' for p in paths)  # noqa: E731
    lit = lambda paths: " ".join(f'(literal "{p}")' for p in paths)  # noqa: E731
    return "\n".join([
        "(version 1)",
        "(deny default)",
        f"(allow process-exec {lit(exe)})",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        f"(allow file-read* file-map-executable {sub([run, RUNTIME, *python, *SYSTEM_READS])} {lit(DEVICE_READS)} (literal \"/\"))",
        f'(allow file-write* (subpath "{run}") {lit(["/dev/null", "/dev/dtracehelper"])})',
        f'(allow ipc-posix-shm* (ipc-posix-name-prefix "{SHM_PREFIX}"))',
        '(allow network* (local ip "localhost:*") (remote ip "localhost:*"))',
        "(allow mach-lookup " + " ".join(f'(global-name "{n}")' for n in MACH_LOOKUPS) + ")",
        '(deny file-read* file-write* (regex #"/_private(/|$)"))',
    ])


def wrap(argv: list[str], run_dir: Path, allow_unsandboxed: bool = False) -> list[str]:
    """argv wrapped to run sandboxed. Raises SandboxUnavailable if it cannot be, unless the caller explicitly opted out."""
    if available():
        return [SANDBOX_EXEC, "-p", profile(run_dir), *argv]
    if allow_unsandboxed:
        return argv
    raise SandboxUnavailable("job isolation was requested but sandbox-exec cannot be applied here; refusing to run "
                             "the job unsandboxed (opt out explicitly with allow_unsandboxed=True for repo-owned code)")
