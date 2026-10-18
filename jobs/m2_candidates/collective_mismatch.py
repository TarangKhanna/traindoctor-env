"""COLLECTIVE_MISMATCH: from a seeded step one rank enters a different collective than its peers,
so the others time out. Variants: bucket-0 broadcast, last-bucket reduce."""
import random

from jobs.config import save_config
from jobs.faults import FaultSpec, base_configs

LABEL = "COLLECTIVE_MISMATCH"


def build(seed: int, variant: int, world_size: int) -> FaultSpec:
    rng = random.Random(f"{LABEL}:{seed}:{variant}")
    prev, cfg = base_configs(rng, seed, world_size)
    rank, step = rng.randrange(world_size), rng.choice([5, 6, 7])
    sync = ("broadcast", "reduce")[variant]
    cfg.rank_overrides.setdefault(str(rank), {}).update(grad_sync=sync, grad_sync_from_step=step)
    return FaultSpec(LABEL, variant, cfg, prev, lambda jd: save_config(prev, jd / "config.json"), None,
                     "COLLECTIVE_TIMEOUT",
                     dict(faulty_rank=rank, fault_step=step, bad_op=sync,
                          fix_key=f"rank_overrides.{rank}.grad_sync", fix_value="all_reduce"),
                     ref_cfg=prev)
