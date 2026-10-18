"""Seed splits. DEV = reserved pilot range (model selection, frozen-vs-trained comparison). TRAIN = disjoint range used for
SFT/GRPO. TEST = SEALED: never generated, opened, or iterated by any script; only this constant exists (empty in the public release)."""
import os

from traindoctor_env.pilot import DEV_SEEDS

TRAIN_SEEDS = range(20_000, 20_400)
# The sealed TEST seed pool is not published (it was used once; releasing it would let it leak into training).
# A private run supplies it as TD_TEST_SEED_RANGE="lo:hi"; unset -> empty, so no public script can generate TEST episodes.
_lo, _, _hi = os.environ.get("TD_TEST_SEED_RANGE", "0:0").partition(":")
TEST_SEEDS_SEALED = range(int(_lo), int(_hi or _lo))
BURNED_SEEDS = range(7001, 7009)   # seeds already spent in earlier sessions; must never train


def _disjoint(*rs):
    seen = set()
    for r in rs:
        assert seen.isdisjoint(r), "seed ranges overlap"
        seen |= set(r)


_disjoint(DEV_SEEDS, TRAIN_SEEDS, TEST_SEEDS_SEALED, BURNED_SEEDS)
