"""Source-only Harbor adapter; unsupported execution isolation is an error, never reward zero."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from jobs import safefs, sandbox
from traindoctor_env.reexec import Replay
from traindoctor_env.snapshot import Snapshot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker_bridge import evaluate_replay, root_owned

MAX_SOURCE_BYTES = 64 << 10


class InvalidSubmission(ValueError):
    pass


def read_submission(root: Path) -> str:
    """Parse bounded source as data. Never import or execute the submitted file here."""
    try:
        if set(safefs.listdir(root)) != {"ckpt.py"}:
            raise InvalidSubmission("submission must contain only ckpt.py")
        source = safefs.read_bytes(root, "ckpt.py", MAX_SOURCE_BYTES).decode("utf-8")
        compile(source, "submitted-ckpt.py", "exec")
        return source
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        raise InvalidSubmission("invalid source artifact: " + type(exc).__name__) from exc


def evaluate(source: str, fixture: Path, output: Path) -> dict:
    os.environ.setdefault("TD_KEY", os.urandom(32).hex())  # ephemeral trusted verifier data key
    if sys.platform != "linux" and not sandbox.available():
        raise sandbox.SandboxUnavailable("existing local rank isolation is unavailable")
    snap = Snapshot.load(fixture)
    replay = Replay(snap, allow_unsandboxed=False, format_check="v1.2")
    try:
        if not replay.write_file("job/ckpt.py", source):
            raise InvalidSubmission("source could not be staged into the declared job file")
        if sys.platform == "linux":
            outcome = evaluate_replay(replay, fixture, output)
            if not outcome.get("sandboxed") or not outcome.get("worker_isolation"):
                raise sandbox.SandboxUnavailable("worker isolation was not established")
            return outcome
        outcome = replay.outcome()
        if not outcome.sandboxed:
            raise sandbox.SandboxUnavailable("execution returned without verified rank isolation")
        return asdict(outcome)
    finally:
        replay.close()


def run(submission: Path, out: Path, fixture: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    safefs.listdir(out)  # reject a symlink output root before changing permissions
    if sys.platform == "linux" and os.geteuid() == 0:
        root_owned(out.parent)  # agent must not be able to rename the protected output directory
        os.chown(out, 0, 0)
        out.chmod(0o700)
    # The output is verifier-owned, never transferred from the agent. Remove stale rewards.
    for name in ("reward.json", "reward.txt"):
        (out / name).unlink(missing_ok=True)
    result = {"source": "executed", "protocol": "harbor-training-repair-01-v1",
              "verdict": None, "execution_performed": False, "terminal_reward": None}
    code = 0
    try:
        source = read_submission(submission)
        result["submitted_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        result["execution_performed"] = None  # an exception may interrupt execution before a verdict exists
        outcome = evaluate(source, fixture, out)
        result.update(verdict=outcome, execution_performed=True,
                      terminal_reward=int(outcome["ok"]), status="graded")
    except InvalidSubmission as exc:
        result.update(terminal_reward=0, execution_performed=False, status="invalid_submission", reason=str(exc))
    except Exception as exc:
        result.update(status="infrastructure_error", reason=str(exc), error_type=type(exc).__name__)
        code = 2
    safefs.write_bytes(out, "grade_result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    if result["terminal_reward"] is not None:
        safefs.write_bytes(out, "reward.json", (json.dumps({"reward": result["terminal_reward"]}) + "\n").encode())
    print(json.dumps(result, sort_keys=True))
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("/submission"))
    parser.add_argument("--out", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--fixture", type=Path, default=Path("/opt/traindoctor-fixture"))
    args = parser.parse_args()
    return run(args.submission, args.out, args.fixture)


if __name__ == "__main__":
    raise SystemExit(main())
