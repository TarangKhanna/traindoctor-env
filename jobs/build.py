"""A scenario = the pristine job template + a seeded source/config patch that introduces the bug.
The patch is a list of textual Edits; the repair is the same edits reversed. Nothing in the materialized job
says which edit was applied."""
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from jobs import safefs
from jobs.config import dumps, loads, validate_config
from jobs.template import config as _  # noqa: F401  (template must be importable)

TEMPLATE = Path(__file__).parent / "template"
KINDS = ["A1", "A2", "B_SKIP", "B_DUP", "B_OK"]
LABELS = {"A1": "COLLECTIVE_DIVERGENCE", "A2": "RANK_STALLED_BEFORE_COLLECTIVE",
          "B_SKIP": "CKPT_SAMPLER_INCONSISTENT", "B_DUP": "CKPT_SAMPLER_INCONSISTENT", "B_OK": "NO_FAULT",
          "A2v2": "RANK_STALLED_BEFORE_COLLECTIVE"}   # A2v2: scenario version 2 (jobs/a2v2.py), not in KINDS
ALL_LABELS = sorted(set(LABELS.values()))
STEPS, CKPT_EVERY = 12, 4
# The only files whose changes the verifier takes from an agent. Everything else under job/ (the config parser)
# is part of the harness's trusted surface; the launcher, instrumentation and checks live outside the workspace.
EDITABLE = ("job/train.py", "job/data.py", "job/ckpt.py", "job/config.yaml")


@dataclass(frozen=True)
class Edit:
    path: str   # relative to the job directory, e.g. "job/train.py"
    old: str
    new: str

    def reversed(self) -> "Edit":
        return Edit(self.path, self.new, self.old)


@dataclass
class JobSpec:
    kind: str
    variant: int
    config: dict                    # pristine config.yaml content
    bug: list[Edit]                 # applied to the pristine job
    kill: tuple[int, int] | None    # (rank, step) worker failure for the incident (Task B), else None
    details: dict = field(default_factory=dict)

    @property
    def fix(self) -> list[Edit]:
        return [e.reversed() for e in reversed(self.bug)]

    @property
    def task(self) -> str:
        return self.kind[0]


def base_config(world: int, job_seed: int, warmup: int, shard_steps: int, decoy_rank: int) -> dict:
    return dict(
        world_size=world, steps=STEPS, seed=job_seed, hidden=12, lr=0.1, batch_size=4, ckpt_every=CKPT_EVERY,
        step_time_s=0.05, pg_timeout_s=3, watchdog_s=8,
        sync=dict(mode="all_reduce", warmup_steps=warmup, per_rank={str(decoy_rank): "all_reduce"}),
        data=dict(root="data", num_samples=96, shard_steps=shard_steps,
                  shard_dirs={str(r): f"part-{r}" for r in range(world)}),
    )


def build(kind: str, variant: int, world: int, job_seed: int, rng: random.Random) -> JobSpec:
    """All draws are made up front, in the same order for every kind, so the public numbers (steps, ranks)
    follow one distribution regardless of kind."""
    if kind == "A2v2":
        from jobs import a2v2
        return a2v2.build(variant, world, job_seed, rng)
    f = rng.choice([5, 6, 7])           # step at which a Task A job goes wrong
    rank = rng.randrange(world)
    decoy_warmup, decoy_shard = rng.choice([5, 6, 7]), rng.choice([5, 6, 7])
    decoy_rank = rng.randrange(world)
    kill_step = rng.choice([5, 6, 9, 10])
    a1_cfg = kind == "A1" and variant == 1
    cfg = base_config(world, job_seed, f if a1_cfg else decoy_warmup, f if kind == "A2" else decoy_shard,
                      rank if a1_cfg else decoy_rank)
    bug: list[Edit] = []
    kill = None
    details: dict = dict(rank=rank)
    if kind == "A1":
        details.update(step=f, bad_op="reduce" if variant else "broadcast")
        if variant == 0:  # a leftover rank-keyed branch in the gradient sync
            old = "            comm.all_reduce(bucket, step, b)\n"
            new = (f"            if rank == {rank} and step >= {f} and b == 0:\n"
                   f"                comm.broadcast(bucket, step, b)\n"
                   f"            else:\n"
                   f"                comm.all_reduce(bucket, step, b)\n")
            bug = [Edit("job/train.py", old, new)]
        else:             # a per-rank sync mode (reduce+broadcast) set for one rank only
            bug = [Edit("job/config.yaml", f"    {rank}: all_reduce\n", f"    {rank}: reduce_bcast\n")]
    elif kind == "A2":  # the rank's shard directory is a stale one that only has the first shard
        details.update(step=f)
        bug = [Edit("job/config.yaml", f"    {rank}: part-{rank}\n", f"    {rank}: part-{world}\n")]
    elif kind in ("B_SKIP", "B_DUP", "B_OK"):
        kill = (rank, kill_step)
        details.update(step=kill_step)
        if kind == "B_SKIP":  # restore the data position from the progress file, not from the checkpoint
            bug = [Edit("job/ckpt.py", "def restore(", SKIP_HELPER + "def restore("),
                   Edit("job/ckpt.py", '        sampler.load_state(state["sampler"])\n',
                        '        sampler.load_state(_progress(rank, state["sampler"]))\n')]
        elif kind == "B_DUP":  # the sampler position is not written into the checkpoint
            bug = [Edit("job/ckpt.py", "state.update(sampler=sampler.state(), step=torch.tensor([step]))",
                        "state.update(step=torch.tensor([step]))")]
    return JobSpec(kind, variant, cfg, bug, kill, details)


SKIP_HELPER = '''def _progress(rank: int, saved):
    """Data position the previous attempt reached on this rank (falls back to the checkpoint's)."""
    p = Path("progress", f"rank{rank}.json")
    return torch.tensor(json.loads(p.read_text())["sampler"]) if p.exists() else saved


'''


def apply_edits(job_dir: Path, edits: list[Edit]) -> None:
    for e in edits:
        text = safefs.read_bytes(job_dir, e.path).decode()
        if text.count(e.old) != 1:
            raise ValueError(f"edit target not found exactly once in {e.path}")
        safefs.write_bytes(job_dir, e.path, text.replace(e.old, e.new).encode())


def materialize(job_dir: Path, config: dict, bug: list[Edit] = ()) -> None:
    """Fresh job directory: job/ (template + config + optional bug) and the data shards the data service published."""
    job_dir = Path(job_dir)
    shutil.copytree(TEMPLATE, job_dir / "job", ignore=shutil.ignore_patterns("__init__.py", "__pycache__"))
    safefs.write_bytes(job_dir, "job/config.yaml", dumps(config).encode())
    v2 = (config.get("data") or {}).get("producer") is not None   # scenario version 2: shards come from a producer
    if v2:
        from jobs import a2v2
        a2v2.overlay(job_dir)
    apply_edits(job_dir, list(bug))
    cfg = loads(safefs.read_bytes(job_dir, "job/config.yaml").decode())
    validate_config(cfg)
    n_shards = -(-cfg["steps"] // cfg["data"]["shard_steps"])
    root = job_dir / cfg["data"]["root"]
    if v2:
        return
    for r in range(cfg["world_size"]):
        for k in range(n_shards):
            (root / f"part-{r}").mkdir(parents=True, exist_ok=True)
            (root / f"part-{r}" / f"shard{k}.idx").write_text(f'{{"shard": {k}}}\n')
    (root / f"part-{cfg['world_size']}").mkdir(parents=True, exist_ok=True)  # an old directory with only shard 0
    (root / f"part-{cfg['world_size']}" / "shard0.idx").write_text('{"shard": 0}\n')
