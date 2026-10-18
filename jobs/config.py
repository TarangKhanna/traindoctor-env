"""Framework-side access to a job's config.yaml (the parser itself lives with the job source)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jobs import safefs
from jobs.template import config as _yaml

loads, dumps = _yaml.loads, _yaml.dumps


def job_config(job_dir: Path) -> dict:
    return _yaml.loads(safefs.read_bytes(job_dir, "job/config.yaml", 1 << 20).decode())


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out
