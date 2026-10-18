"""Checkpoint manifest: ckpt/MANIFEST.json is a list of entries
{ckpt_id, step, world_size, shards:[{file, sha256, shape}]}; files are relative to ckpt/. The job writes it
(job/ckpt.py); the scheduler side validates it before trusting a checkpoint."""
from __future__ import annotations

import io
import json
from pathlib import Path

import torch

from jobs import safefs

from jobs.template.ckpt import digest


class ManifestError(Exception):
    pass


def path(job_dir: Path) -> Path:
    return Path(job_dir) / "ckpt" / "MANIFEST.json"


def load(job_dir: Path) -> list[dict]:
    try:
        entries = json.loads(safefs.read_bytes(job_dir, "ckpt/MANIFEST.json"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):   # the job writes this file: garbage, a symlink or a FIFO counts as no checkpoints
        return []
    return entries if isinstance(entries, list) and all(isinstance(e, dict) for e in entries) else []


def check_entry(job_dir: Path, e: dict) -> None:
    """Raise ManifestError if the entry's files are missing or do not match its checksums."""
    try:
        for s in e["shards"]:
            try:
                blob = safefs.read_bytes(job_dir, "ckpt/" + s["file"])
            except (OSError, ValueError):
                raise ManifestError(f"{e['ckpt_id']}: missing file {s['file']}") from None
            if digest(torch.load(io.BytesIO(blob), weights_only=True)) != s["sha256"]:
                raise ManifestError(f"{e['ckpt_id']}: checksum mismatch for {s['file']}")
    except ManifestError:
        raise
    except Exception as exc:  # noqa: BLE001 -- the entry is job-written; any malformation means "not valid"
        raise ManifestError(f"malformed manifest entry: {type(exc).__name__}") from None


def validate(job_dir: Path, entries: list[dict]) -> None:
    """Unique ckpt_id, strictly increasing step, every entry consistent with its files."""
    seen: set[str] = set()
    prev = -1
    for e in entries:
        if e["ckpt_id"] in seen:
            raise ManifestError(f"duplicate ckpt_id {e['ckpt_id']} in manifest")
        seen.add(e["ckpt_id"])
        if e["step"] <= prev:
            raise ManifestError(f"non-monotonic step at {e['ckpt_id']}: {e['step']} after {prev}")
        prev = e["step"]
        check_entry(job_dir, e)


def latest_valid(job_dir: Path, entries: list[dict]) -> dict | None:
    for e in reversed(entries):
        try:
            check_entry(job_dir, e)
            return e
        except ManifestError:
            continue
    return None
