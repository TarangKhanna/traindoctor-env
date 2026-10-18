"""State that belongs to the launcher, not to the job. For a job directory D it lives in the sibling directory
D.trusted (outside everything a job process may write or, under the sandbox, read):

  ledger.jsonl          one record per run: how it was launched, how it ended, what the data server served,
                        and the checkpoints the launcher captured while the run was alive
  captures/<run>/*.pt   byte copies of every checkpoint at the time the launcher saw it committed, hashed

Lost work, relaunch counts and checkpoint equivalence are all judged from here, never from files the job writes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from jobs import safefs

PRIVATE = "_private"


def trusted_dir(job_dir: Path) -> Path:
    job_dir = Path(job_dir)
    return job_dir.with_name(job_dir.name + ".trusted")


def ledger(job_dir: Path) -> list[dict]:
    p = trusted_dir(job_dir) / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def append_run(job_dir: Path, rec: dict) -> None:
    t = trusted_dir(job_dir)
    t.mkdir(parents=True, exist_ok=True)
    with open(t / "ledger.jsonl", "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def append_event(job_dir: Path, kind: str, detail: str) -> None:
    """Scheduler events live in trusted storage, never in a file of the job-writable tree."""
    _trusted_write(job_dir, "scheduler.jsonl", (json.dumps(dict(event=kind, detail=detail), sort_keys=True) + "\n").encode(), append=True)


def events(job_dir: Path) -> list[dict]:
    p = trusted_dir(job_dir) / "scheduler.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def run_ids(job_dir: Path) -> list[str]:
    return [r["run_id"] for r in ledger(job_dir)]


def copy_job(src: Path, dst: Path) -> None:
    """A working copy of a job directory (and its trusted ledger). Private material never travels: a copy that
    a job process can run in must not contain the reference, the metadata or anything under _private."""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(PRIVATE))
    if trusted_dir(src).exists():
        shutil.copytree(trusted_dir(src), trusted_dir(dst))


RUN_ID_RE = re.compile(r"^run[0-9]{1,6}$")


def _check_rel(rel: str) -> list[str]:
    return safefs.check_rel(rel)


def read_nofollow(base: Path, rel: str, max_bytes: int = 64 << 20) -> bytes:
    """Read base/rel from a tree a job process can write to (see jobs/safefs.py). `base` is the pinned root and is
    itself opened without following a symlink; pass the trusted job directory, never a path below it that the
    job controls (e.g. read "ckpt/x", not base=job/ckpt)."""
    return safefs.read_bytes(base, rel, max_bytes)


def _trusted_write(job_dir: Path, rel: str, data: bytes, append: bool = False) -> Path:
    """Write under trusted_dir(job_dir). The destination must resolve inside it (asserted), whatever `rel` says."""
    root = trusted_dir(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    root_r = root.resolve()
    out = root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    assert root_r in out.parent.resolve().parents or out.parent.resolve() == root_r, "trusted write escapes trusted_dir"
    assert not out.is_symlink(), "trusted write target is a symlink"
    with open(out, "ab" if append else "wb") as f:
        f.write(data)
    return out


def write_run_record(job_dir: Path, run_id: str, name: str, data: str) -> Path:
    """Launcher-owned per-run records (job.json, samples/rankN.jsonl): trusted_dir/records/<run_id>/<name>, never
    a path inside the job-writable run directory."""
    assert RUN_ID_RE.match(run_id), "bad run id"
    _check_rel(name)
    return _trusted_write(job_dir, f"records/{run_id}/{name}", data.encode())


def read_run_record(job_dir: Path, run_id: str, name: str) -> str | None:
    p = trusted_dir(job_dir) / "records" / run_id / name
    return p.read_text() if p.exists() else None


class Capturer:
    """Copies checkpoints the moment the launcher sees them in the job's manifest. Called while rank 0 is parked
    in a data request (checkpoint written, barrier passed) and once more after every process of the run is dead."""

    def __init__(self, job_dir: Path, run_id: str):
        self.job_dir, self.run_id = Path(job_dir), run_id
        self.records: list[dict] = []
        self.invalid: list[str] = []
        self._seen: set[tuple] = set()

    def scan(self) -> None:
        """Trusted file names come from a trusted counter and the content hash only. ckpt_id/step are job-supplied
        and are stored as data in the record, never in a path. The shard is read from the job dir with a
        no-symlink component walk and must stay under it; anything else is recorded in self.invalid."""
        try:
            entries = json.loads(read_nofollow(self.job_dir, "ckpt/MANIFEST.json"))
            assert isinstance(entries, list)
        except (OSError, ValueError, AssertionError):
            return
        for e in entries:
            try:
                cid, step, rel = str(e["ckpt_id"]), int(e["step"]), e["shards"][0]["file"]
                blob = read_nofollow(self.job_dir, "ckpt/" + rel)
            except (KeyError, IndexError, TypeError, ValueError, OSError):
                self.invalid.append("unreadable or out-of-tree checkpoint entry")
                continue
            sha = hashlib.sha256(blob).hexdigest()
            if (cid, step, sha) in self._seen:
                continue
            self._seen.add((cid, step, sha))
            name = f"captures/{self.run_id}/{len(self.records):06d}.{sha[:24]}.pt"
            out = _trusted_write(self.job_dir, name, blob)
            self.records.append(dict(ckpt_id=cid, step=step, sha256=sha,
                                     file=str(out.relative_to(trusted_dir(self.job_dir)))))
