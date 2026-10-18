"""Pilot harness skeleton: a few stratified DEV episodes, k attempts each, any `generate(prompt) -> (text, n_tokens)`
callable (an untrained Qwen via mlx-lm lives outside the core). Records what decides whether RL has signal."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from jobs.build import KINDS
from traindoctor_env.bundle import make_bundle, make_menu, outcome_table
from traindoctor_env.scenario import make_scenario
from traindoctor_env.snapshot import execute, load_truth
from traindoctor_env.tiera import parse_answer, reward, training_prompt

DEV_SEEDS = range(10_000, 10_400)     # a seed range reserved for pilots; never used for training


def stratified_seeds(n: int, pool=DEV_SEEDS) -> list[int]:
    """n seeds, round-robin over the scenario kinds (counts differ by at most one)."""
    by_kind = defaultdict(list)
    for s in pool:
        by_kind[make_scenario(s).kind].append(s)
    return [by_kind[KINDS[i % len(KINDS)]][i // len(KINDS)] for i in range(n)]


def prepare(seed: int, workdir):
    """Real execution of one scenario + its fix-outcome table (slow)."""
    snap = execute(make_scenario(seed), workdir / str(seed))
    truth = load_truth(snap)
    menu = make_menu(snap, truth)
    return truth, make_bundle(snap, menu, truth), menu, outcome_table(snap, menu)


def run_pilot(episodes: list, generate: Callable[[str], tuple[str, int]], attempts: int = 4) -> dict:
    """episodes: [(truth, bundle, menu, table)]. Per-episode and overall rates plus reward diversity, where an
    episode is mixed when its repeated attempts include both reward 0 and reward 1 (V2)."""
    rows = []
    for truth, bundle, menu, table in episodes:
        prompt, ids = training_prompt(bundle), {c.id for c in menu}
        rs = []
        for _ in range(attempts):
            t0 = time.monotonic()
            text, n_tok = generate(prompt)
            r = reward(text, truth, bundle, menu, table, "V2")
            a = parse_answer(text, ids)
            rs.append(dict(seconds=time.monotonic() - t0, tokens=n_tok, valid=r.valid, unsafe=r.unsafe,
                           diagnosis=r.v0, recovery=r.v2, reward_v0=float(r.v0), reward_v2=r.reward,
                           repair=a.repair if a else None))
        rows.append(dict(seed=truth.seed, kind=truth.kind, attempts=rs,
                         mixed_v2=len({x["reward_v2"] for x in rs}) > 1))
    flat = [a for r in rows for a in r["attempts"]]
    n = len(flat) or 1
    rate = lambda k: sum(bool(a[k]) for a in flat) / n  # noqa: E731
    return dict(episodes=len(rows), attempts=len(flat), recovery=rate("recovery"), diagnosis=rate("diagnosis"),
                invalid=1 - rate("valid"), unsafe=rate("unsafe"), mean_seconds=sum(a["seconds"] for a in flat) / n,
                mean_tokens=sum(a["tokens"] for a in flat) / n,
                mixed_episodes=sum(r["mixed_v2"] for r in rows), rows=rows)
