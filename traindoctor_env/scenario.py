"""Scenario = what to execute; Truth = what was injected (never part of a snapshot's public data)."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jobs.build import ALL_LABELS, KINDS, LABELS, JobSpec, build

__all__ = ["ALL_LABELS", "KINDS", "LABELS", "Scenario", "Truth", "job_spec", "make_scenario"]


@dataclass(frozen=True)
class Scenario:
    seed: int        # generator-level id; chooses everything below through private streams
    kind: str        # A1 | A2 | B_SKIP | B_DUP | B_OK
    variant: int
    world_size: int
    job_seed: int    # the training seed shown in the job config; independent of kind/variant

    @property
    def label(self) -> str:
        return LABELS[self.kind]


@dataclass
class Truth:
    seed: int
    kind: str
    variant: int
    label: str
    task: str
    fix: list = field(default_factory=list)   # [{path, old, new}] repairing the job source/config
    details: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Truth":
        return cls(**json.loads(Path(path).read_text()))


def make_scenario(seed: int) -> Scenario:
    """Kind, variant, world size and the job's own seed come from separate private streams, so no public
    value (seed, ids, sizes) is a function of the label."""
    rng = random.Random(f"scenario:{seed}")
    kind = rng.choice(KINDS)
    return Scenario(seed, kind, rng.randrange(2) if kind == "A1" else 0, rng.choice([3, 4]),
                    random.Random(f"job-seed:{seed}").randrange(1, 10**6))


def job_spec(sc: Scenario) -> JobSpec:
    return build(sc.kind, sc.variant, sc.world_size, sc.job_seed, random.Random(f"build:{sc.seed}"))
