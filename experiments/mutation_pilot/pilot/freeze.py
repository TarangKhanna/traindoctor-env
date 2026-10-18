"""Freeze the 8 pilot tasks before any agent runs: writes tasks.json (hashes of the mutant definitions, the buggy job
files, the incident records, the validation receipts and the canonical code commit) and prints the freeze hash."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot import core  # noqa: E402

sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()  # noqa: E731


def main() -> None:
    head = subprocess.run(["git", "-C", str(core.CANON), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                          env={"GIT_CONFIG_GLOBAL": "/dev/null", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}).stdout.strip()
    tasks = {}
    for tid in sorted(core.TASKS):
        snap = core.build_snapshot(tid)
        m = core.mutant(tid)
        rec = json.loads((core.HERE / "receipts" / f"{tid}.json").read_text())
        assert rec["all_checks_pass"], f"{tid} failed validation; do not freeze"
        tasks[tid] = dict(seed=m.seed, operator=m.op, variant=m.variant, mutant_sha256=m.digest(),
                          buggy_files_sha256={k: sha(v) for k, v in sorted(snap.files.items())},
                          records_sha256=sha("".join(json.dumps(r, sort_keys=True) for r in snap.records)),
                          receipt_sha256=sha((core.HERE / "receipts" / f"{tid}.json").read_bytes()),
                          mechanism_fingerprint=rec["fingerprint"]["mechanism_hash"], full_fingerprint=rec["fingerprint"]["full_hash"])
    body = dict(canonical_code=dict(branch="modal-infra", head=head), seeds=[m["seed"] for m in tasks.values()], tasks=tasks)
    body["freeze_sha256"] = sha(json.dumps(body, sort_keys=True))
    (core.HERE / "tasks.json").write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("freeze_sha256", body["freeze_sha256"], "modal-infra", head[:10])


main()
