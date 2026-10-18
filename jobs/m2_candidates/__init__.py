"""Executable faults. Each module exposes build(seed, variant, world_size) -> FaultSpec."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jobs.config import JobConfig


@dataclass
class FaultSpec:
    label: str
    variant: int
    cfg: JobConfig                      # config in force at the failing launch
    prev_cfg: JobConfig                 # config of the previous (healthy) launch
    prepare_fn: Callable[[Path], None]  # sets up pre-existing job_dir state, incl. earlier runs
    resume_from: str | None             # checkpoint of the failing launch
    failure_class: str                  # canonical class the failing launch must end with
    details: dict = field(default_factory=dict)  # ground truth specifics (-> truth.json)
    ref_cfg: JobConfig | None = None    # fault-free config for the uninterrupted reference run


def base_configs(rng: random.Random, seed: int, world_size: int) -> tuple[JobConfig, JobConfig]:
    """(prev, cfg) differing only by harmless per-rank overrides: red herrings in the config diff."""
    ov: dict[str, dict] = {}
    for r in rng.sample(range(world_size), k=min(2, world_size)):
        ov[str(r)] = {rng.choice(["log_level", "profile", "dataloader_prefetch"]): None}
    vals = {"log_level": ("info", "debug"), "profile": (False, True), "dataloader_prefetch": (2, 4)}
    prev, cur = {}, {}
    for r, d in ov.items():
        (k,) = d
        prev[r], cur[r] = {k: vals[k][0]}, {k: vals[k][1]}
    base = dict(world_size=world_size, seed=seed)
    return (JobConfig(**base, rank_overrides=prev), JobConfig(**base, rank_overrides=cur))


def registry() -> dict:
    from jobs.faults import collective_mismatch, nonidempotent_resubmit, preemption_no_cleanup, sharding_misconfig
    return {m.LABEL: m for m in (collective_mismatch, sharding_misconfig, preemption_no_cleanup, nonidempotent_resubmit)}
