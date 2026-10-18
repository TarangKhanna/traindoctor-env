"""PREEMPTION_NO_CLEANUP: the preempted run leaves its .lock behind and resubmission is refused.
Variants: preempted right after a checkpoint / two steps after it."""
import random

from jobs.config import save_config
from jobs.faults import FaultSpec, base_configs
from jobs.resubmit import resubmit

LABEL = "PREEMPTION_NO_CLEANUP"


def build(seed: int, variant: int, world_size: int) -> FaultSpec:
    rng = random.Random(f"{LABEL}:{seed}:{variant}")
    _, cfg = base_configs(rng, seed, world_size)
    after = (8, 10)[variant]

    def prepare(jd):
        save_config(cfg, jd / "config.json")
        assert resubmit(jd, preempt_after=after).status == "preempted"

    return FaultSpec(LABEL, variant, cfg, cfg, prepare, None, "LOCK_HELD", dict(preempted_after=after, latest_ckpt="ck0008"), ref_cfg=cfg)
