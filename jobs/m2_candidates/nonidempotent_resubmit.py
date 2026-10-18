"""NONIDEMPOTENT_RESUBMIT: the scheduler force-clears the lock of each preempted run and resubmits
without an idempotency key. The replay re-commits checkpoints, so the manifest gets duplicate entries
and the next resume fails validation. Variants: how far the two preempted runs got."""
import random

from jobs.config import save_config
from jobs.faults import FaultSpec, base_configs
from jobs.resubmit import force_clear_lock, resubmit

LABEL = "NONIDEMPOTENT_RESUBMIT"


def build(seed: int, variant: int, world_size: int) -> FaultSpec:
    rng = random.Random(f"{LABEL}:{seed}:{variant}")
    _, cfg = base_configs(rng, seed, world_size)
    first, second = ((10, 10), (8, 9))[variant]

    def prepare(jd):
        save_config(cfg.model_copy(update=dict(steps=4)), jd / "config.json")
        assert resubmit(jd).ok                                       # run0: ck0004
        save_config(cfg, jd / "config.json")
        assert resubmit(jd, resume_from="ck0004", preempt_after=first).status == "preempted"
        force_clear_lock(jd)
        assert resubmit(jd, preempt_after=second).status == "preempted"  # replays from ck0004, duplicates ck0008
        force_clear_lock(jd)

    return FaultSpec(LABEL, variant, cfg, cfg, prepare, None, "MANIFEST_INVALID",
                     dict(duplicated_ckpt="ck0008", latest_ckpt="ck0008"), ref_cfg=cfg)
