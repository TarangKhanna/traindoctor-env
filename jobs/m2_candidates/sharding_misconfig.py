"""SHARDING_MISCONFIG: a job writes split checkpoint shards, then is restarted with a different
world size, so loading fails with a shape error. Variants: shrink 4->2, grow 2->4."""
import random

from jobs.config import save_config
from jobs.faults import FaultSpec, base_configs
from jobs.resubmit import resubmit

LABEL = "SHARDING_MISCONFIG"


def build(seed: int, variant: int, world_size: int) -> FaultSpec:
    ckpt_ws, new_ws = ((4, 2), (2, 4))[variant]
    rng = random.Random(f"{LABEL}:{seed}:{variant}")
    prev, cfg = base_configs(rng, seed, ckpt_ws)
    prev = prev.model_copy(update=dict(shard_layout="split", steps=8))
    cfg = cfg.model_copy(update=dict(shard_layout="split", world_size=new_ws))

    def prepare(jd):
        save_config(prev, jd / "config.json")
        assert resubmit(jd).ok

    return FaultSpec(LABEL, variant, cfg, prev, prepare, "ck0008", "SHAPE_ERROR",
                     dict(ckpt_world_size=ckpt_ws, new_world_size=new_ws, latest_ckpt="ck0008",
                          fix_key="world_size", fix_value=ckpt_ws),
                     ref_cfg=prev.model_copy(update=dict(steps=12)))
