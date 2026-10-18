"""Task A2, scenario version 2 (kind "A2v2"): the stalled rank waits on a shard that a real background producer is slow
to publish.

In version 1 the stall is the shard wait itself, so deleting the wait removes the cause. Here the wait is a readiness
protocol (index file first, `.ready` marker last) that every rank needs: consumers that skip it read the producer's
preallocated placeholder index and fetch the wrong batches, which sample accounting and the state checks reject. The
injected fault is one rank's producer being throttled; the repair is on the producer side (rate, staging, ordering).
Version 1 (kind "A2") is untouched; this module is reached only through kind "A2v2"."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from jobs import safefs

TEMPLATE_V2 = Path(__file__).parent / "template_a2v2"
KIND = "A2v2"
LABEL = "RANK_STALLED_BEFORE_COLLECTIVE"
SHARD_MB, HEALTHY_MB_S, DECOY_MB_S, STALLED_MB_S = 4, 400, 3, 0.01   # lags of 0.01 s, 1.3 s (< pg timeout), 400 s


def is_v2(config: dict) -> bool:
    return isinstance((config.get("data") or {}).get("producer"), dict)


def build(variant: int, world: int, job_seed: int, rng: random.Random):
    from jobs.build import Edit, JobSpec, base_config
    f = rng.choice([5, 6, 7])           # same draw order as jobs.build.build, so (seed, world) give the same rank/step as A2
    rank = rng.randrange(world)
    decoy_warmup, _ = rng.choice([5, 6, 7]), rng.choice([5, 6, 7])
    decoy_rank = rng.randrange(world)
    rng.choice([5, 6, 7, 9, 10])
    cfg = base_config(world, job_seed, decoy_warmup, f, decoy_rank)
    slow = (rank + 1) % world           # a producer that is slow but fine: the red herring
    cfg["data"]["producer"] = dict(shard_mb=SHARD_MB, cached_shards=1,
                                   throttle_mb_s={str(r): (DECOY_MB_S if r == slow and world > 2 else HEALTHY_MB_S) for r in range(world)})
    old = f"    {rank}: {HEALTHY_MB_S}\n"
    new = f"    {rank}: {STALLED_MB_S}\n"
    return JobSpec(KIND, 0, cfg, [Edit("job/config.yaml", old, new)], None, dict(rank=rank, step=f))


def overlay(job_dir: Path) -> None:
    """Replace the template's data module and training loop with the version-2 ones (called by jobs.build.materialize)."""
    for name in ("data.py", "train.py"):
        safefs.write_bytes(job_dir, f"job/{name}", (TEMPLATE_V2 / name).read_bytes())
