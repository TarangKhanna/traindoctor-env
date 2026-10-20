"""Framework-side access to a job's config.yaml (the parser itself lives with the job source)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from jobs import safefs
from jobs.template import config as _yaml

loads, dumps = _yaml.loads, _yaml.dumps

MAX_WORLD = 16


class BadConfig(ValueError):
    """Agent-editable configuration outside the launcher's resource envelope."""


def bounded(value, name: str, lower, upper, integer: bool = False) -> None:
    types = (int,) if integer else (int, float)
    if type(value) not in types or not lower <= value <= upper or not math.isfinite(value):
        raise BadConfig(f"{name} must be {'an integer' if integer else 'finite'} in [{lower}, {upper}]")


def validate_data_params(world, num_samples, batch_size, seed) -> None:
    for name, value, lower, upper in (
        ("world_size", world, 1, MAX_WORLD), ("num_samples", num_samples, 1, 4096),
        ("batch_size", batch_size, 1, 256), ("seed", seed, 0, 2**31 - 1),
    ):
        bounded(value, name, lower, upper, integer=True)
    if num_samples // world // batch_size < 1:
        raise BadConfig("num_samples must provide at least one full batch per rank")


def validate_config(cfg: dict) -> None:
    """Validate before constructing shards, tensors, or worker processes; never coerce agent values."""
    try:
        validate_data_params(cfg["world_size"], cfg["data"]["num_samples"], cfg["batch_size"], cfg["seed"])
        for name, upper in (("steps", 1000), ("ckpt_every", 1000), ("hidden", 1024)):
            bounded(cfg[name], name, 1, upper, integer=True)
        bounded(cfg["data"]["shard_steps"], "shard_steps", 1, 1000, integer=True)
        bounded(cfg["sync"]["warmup_steps"], "warmup_steps", 0, 1000, integer=True)
        for name in ("pg_timeout_s", "watchdog_s"):
            bounded(cfg[name], name, 0.001, 120)
        bounded(cfg["step_time_s"], "step_time_s", 0, 1)
        bounded(cfg["lr"], "lr", 1e-8, 10)
        if "producer" in cfg["data"]:
            prod = cfg["data"]["producer"]
            bounded(prod["shard_mb"], "shard_mb", 0.001, 1024)
            bounded(prod["cached_shards"], "cached_shards", 0, 1000, integer=True)
            for rank in range(cfg["world_size"]):
                bounded(prod["throttle_mb_s"][str(rank)], "throttle_mb_s", 0.001, 1e6)
    except (KeyError, TypeError, AttributeError) as e:
        raise BadConfig(f"unreadable job config: {type(e).__name__}") from None


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
