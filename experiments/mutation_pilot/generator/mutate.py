"""Procedural mutation operators over the known-good TrainDoctor Gloo job (pilot; seeds 40000-40099 only).

Each operator is a seeded textual patch template over the pristine job template (jobs/template, read-only, from the
canonical modal-infra worktree). A mutant carries:
  bug    the patch that breaks the job (list[Edit], each `old` occurs exactly once when applied in order)
  gold   the reverse patch (restores the pristine bytes)
  alt    a textually different patch that is also a correct repair (None where the frozen verifier admits none)
  decoy  a plausible repair that lets the job COMPLETE but must still fail the recovery invariants
Alt and decoy are expressed against the BUGGY text. No language model writes any part of a mutant.

Operators (task A = relaunch from the incident's checkpoint; task B = fresh run + worker failure + automatic restart):
  O1  on rank k, swap the collective type (a) or the tensor shape (b) of one gradient bucket from step f on   [A]
  O2  checkpoint sampler state: dropped (a) or saved one batch behind the step counter (b)                    [B]
  O3  non-atomic checkpoint-manifest write handed to a background thread, victim = rank 0 killed mid-write:
      the restart reads a torn manifest and either starts from scratch (a) or crashes on it (b)               [B]
  O4  off-by-one in the resume step: +1 in ckpt.restore (a) or -1 at the call site in train.py (b)           [B]
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path



def _repo() -> Path:
    """Checkout that provides jobs/ and traindoctor_env/ (read-only): $TD_REPO, else the nearest parent containing jobs/build.py."""
    if os.environ.get("TD_REPO"):
        return Path(os.environ["TD_REPO"])
    for p in Path(__file__).resolve().parents:
        if (p / "jobs" / "build.py").exists():
            return p
    raise RuntimeError("set TD_REPO to a checkout containing jobs/ and traindoctor_env/")


CANON = _repo()
if str(CANON) not in sys.path:
    sys.path.insert(0, str(CANON))

from jobs.build import Edit, JobSpec, apply_edits, base_config, materialize  # noqa: E402

SEED_RANGE = (40000, 40099)
OPERATORS = ("O1", "O2", "O3", "O4")
VARIANTS = ("a", "b")


@dataclass(frozen=True)
class Mutant:
    seed: int
    op: str
    variant: str
    spec: JobSpec
    alt: tuple | None
    decoy: tuple
    alt_note: str
    decoy_note: str

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def bug(self) -> list:
        return self.spec.bug

    @property
    def gold(self) -> list:
        return self.spec.fix

    def describe(self) -> dict:
        """Everything that defines the mutant, canonical (used for hashing and the byte-determinism test)."""
        e = lambda es: [(x.path, x.old, x.new) for x in es]  # noqa: E731
        return dict(seed=self.seed, op=self.op, variant=self.variant, kind=self.kind, config=self.spec.config,
                    kill=self.spec.kill, details=self.spec.details, bug=e(self.bug), gold=e(self.gold),
                    alt=None if self.alt is None else e(self.alt), decoy=e(self.decoy),
                    alt_note=self.alt_note, decoy_note=self.decoy_note)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- shared text
TRAIN_SYNC_ELSE = "            comm.all_reduce(bucket, step, b)\n"
TRAIN_RESTORE = '        start = ckpt.restore(os.environ["RESUME_FROM"], model, opt, sampler, rank)\n'
CKPT_SAMPLER_SAVE = "state.update(sampler=sampler.state(), step=torch.tensor([step]))"
CKPT_SAMPLER_DROP = "state.update(step=torch.tensor([step]))"
CKPT_RESTORE_SAMPLER = ('    if "sampler" in state:  # checkpoints from before the sampler was tracked have no position\n'
                        '        sampler.load_state(state["sampler"])\n')
CKPT_RETURN_STEP = '    return int(state["step"])\n'
CKPT_COMMIT_BODY = ('    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []\n'
                    '    entries = [e for e in entries if e["step"] < step] + [entry]\n'
                    '    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\\n")\n')

LAG_HELPER = '''def _before_step(sampler):
    """Sampler position as of the start of the step just finished (the step counter is advanced separately)."""
    n = max(sampler.epoch * sampler.steps_per_epoch + sampler.index - 1, 0)
    return torch.tensor(divmod(n, sampler.steps_per_epoch))


'''

PUBLISH_HELPER = '''_writer = None


def _flush() -> None:
    if _writer is not None:
        _writer.join()


def _publish(text: str) -> None:
    """Hand the manifest write to a background thread so the training step is not blocked on the shared filesystem."""
    global _writer

    def run() -> None:
        half = len(text) // 2
        with open(MANIFEST, "w") as f:
            f.write(text[:half])
            f.flush()
            time.sleep(1.0)
            f.write(text[half:])

    _writer = threading.Thread(target=run)
    _writer.start()


'''

COMMIT_TOLERANT = ('    _flush()\n'
                   '    try:\n'
                   '        entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []\n'
                   '    except ValueError:  # unreadable manifest: start a fresh one\n'
                   '        entries = []\n'
                   '    entries = [e for e in entries if e["step"] < step] + [entry]\n'
                   '    _publish(json.dumps(entries, indent=2, sort_keys=True) + "\\n")\n')
COMMIT_STRICT = ('    _flush()\n'
                 '    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []\n'
                 '    entries = [e for e in entries if e["step"] < step] + [entry]\n'
                 '    _publish(json.dumps(entries, indent=2, sort_keys=True) + "\\n")\n')

WRITE_INPLACE = ('        with open(MANIFEST, "w") as f:\n'
                 '            f.write(text[:half])\n'
                 '            f.flush()\n'
                 '            time.sleep(1.0)\n'
                 '            f.write(text[half:])\n')
WRITE_ATOMIC = ('        tmp = MANIFEST.with_suffix(".tmp")\n'
                '        with open(tmp, "w") as f:\n'
                '            f.write(text[:half])\n'
                '            f.flush()\n'
                '            time.sleep(1.0)\n'
                '            f.write(text[half:])\n'
                '        tmp.replace(MANIFEST)\n')
WRITE_FSYNC = ('        with open(MANIFEST, "w") as f:\n'
               '            f.write(text[:half])\n'
               '            f.flush()\n'
               '            os.fsync(f.fileno())\n'
               '            time.sleep(1.0)\n'
               '            f.write(text[half:])\n'
               '            f.flush()\n'
               '            os.fsync(f.fileno())\n')


def _e(path: str, old: str, new: str) -> Edit:
    return Edit(path, old, new)


# ---------------------------------------------------------------------------------------------- operators
def _o1(variant: str, world: int, rng: random.Random) -> tuple:
    f, bucket, rank = rng.choice([5, 6, 7]), rng.choice([0, 1]), rng.randrange(1, world)   # never rank 0 (the root of reduce)
    guard = f"            if rank == {rank} and step >= {f} and b == {bucket}:\n"
    if variant == "a":   # collective type
        inner, otherwise = "                comm.reduce(bucket, step, b)\n", "            else:\n" + "    " + TRAIN_SYNC_ELSE
        bug = [_e("job/train.py", TRAIN_SYNC_ELSE, guard + inner + otherwise)]
        alt = (_e("job/train.py", inner, "                comm.all_reduce(bucket, step, b)\n"),)
        alt_note = "make the rank-keyed branch call all_reduce (keeps the branch, removes the divergence)"
    else:                # tensor shape
        inner, otherwise = "                comm.all_reduce(bucket[1:], step, b)\n", "            else:\n" + "    " + TRAIN_SYNC_ELSE
        bug = [_e("job/train.py", TRAIN_SYNC_ELSE, guard + inner + otherwise)]
        alt = (_e("job/train.py", "bucket[1:]", "bucket"),)
        alt_note = "pass the whole bucket in the rank-keyed branch (keeps the branch, removes the divergence)"
    block = guard + inner + otherwise
    decoy = (_e("job/train.py", block, f"            if b == {bucket}:\n                continue\n" + TRAIN_SYNC_ELSE),)
    decoy_note = f"skip the gradient sync of bucket {bucket} on every rank: the job completes but gradients are never averaged there"
    return bug, alt, alt_note, decoy, decoy_note, None, dict(rank=rank, step=f, bucket=bucket), f, rank


def _o2(variant: str, world: int, rng: random.Random) -> tuple:
    kill_step, rank = rng.choice([5, 6, 9, 10]), rng.randrange(world)
    if variant == "a":   # sampler position not saved
        bug = [_e("job/ckpt.py", CKPT_SAMPLER_SAVE, CKPT_SAMPLER_DROP)]
        alt, alt_note = None, "none: the frozen verifier requires the checkpoint itself to store the sampler position"
        progress = ('    p = Path("progress", f"rank{rank}.json")\n'
                    '    if p.exists():  # the data position the previous attempt reached on this rank\n'
                    '        sampler.load_state(torch.tensor(json.loads(p.read_text())["sampler"]))\n')
        decoy = (_e("job/ckpt.py", CKPT_RESTORE_SAMPLER, progress),)
        decoy_note = "restore the data position from the progress file the previous attempt left, not from the checkpoint"
    else:                # sampler position saved one batch behind the step counter
        bug = [_e("job/ckpt.py", "def save(", LAG_HELPER + "def save("),
               _e("job/ckpt.py", CKPT_SAMPLER_SAVE, "state.update(sampler=_before_step(sampler), step=torch.tensor([step]))")]
        alt = (_e("job/ckpt.py", '    n = max(sampler.epoch * sampler.steps_per_epoch + sampler.index - 1, 0)\n'
                  '    return torch.tensor(divmod(n, sampler.steps_per_epoch))\n', '    return sampler.state()\n'),)
        alt_note = "make the helper return the sampler's current position"
        decoy = (_e("job/ckpt.py", '        sampler.load_state(state["sampler"])\n',
                    '        sampler.load_state(torch.zeros(2, dtype=torch.long))\n'),)
        decoy_note = "reset the data position to the start of the dataset on every restore"
    return bug, alt, alt_note, decoy, decoy_note, (rank, kill_step), dict(rank=rank, step=kill_step), None, None


def _o3(variant: str, world: int, rng: random.Random) -> tuple:
    rank, kill_step = 0, 7                     # the writer (rank 0) is killed right after it hands off the step-8 manifest
    imports = _e("job/ckpt.py", "import hashlib\n", "import hashlib\nimport threading\nimport time\n")
    helpers = _e("job/ckpt.py", "def commit(", PUBLISH_HELPER + "def commit(")
    tolerant = variant == "a"
    body = _e("job/ckpt.py", CKPT_COMMIT_BODY, COMMIT_TOLERANT if tolerant else COMMIT_STRICT)
    bug = [imports, helpers, body]
    alt = (_e("job/ckpt.py", WRITE_INPLACE, WRITE_ATOMIC),)
    alt_note = "keep the background writer but write a temp file and rename it over the manifest"
    if tolerant:
        decoy = (_e("job/ckpt.py", "import threading\n", "import os\nimport threading\n"),
                 _e("job/ckpt.py", WRITE_INPLACE, WRITE_FSYNC))
        decoy_note = "fsync the manifest after each chunk (still written in place, still torn when the writer dies)"
    else:
        decoy = (_e("job/ckpt.py", COMMIT_STRICT, COMMIT_TOLERANT),)
        decoy_note = "make commit() tolerate an unreadable manifest by starting a fresh one"
    return bug, alt, alt_note, decoy, decoy_note, (rank, kill_step), dict(rank=rank, step=kill_step), None, None


def _o4(variant: str, world: int, rng: random.Random) -> tuple:
    kill_step, rank = rng.choice([5, 6, 9, 10]), rng.randrange(world)
    if variant == "a":   # restore() returns the wrong step
        bug = [_e("job/ckpt.py", CKPT_RETURN_STEP, '    return int(state["step"]) + 1\n')]
        alt = (_e("job/train.py", TRAIN_RESTORE, TRAIN_RESTORE[:-1] + " - 1\n"),)
        alt_note = "compensate at the call site in train.py"
        decoy = (_e("job/train.py", TRAIN_RESTORE, "        start = 0\n"),)
    else:                # the call site is off by one
        bug = [_e("job/train.py", TRAIN_RESTORE, TRAIN_RESTORE[:-1] + " - 1\n")]
        alt = (_e("job/ckpt.py", CKPT_RETURN_STEP, '    return int(state["step"]) + 1\n'),)
        alt_note = "compensate inside ckpt.restore"
        decoy = (_e("job/train.py", TRAIN_RESTORE[:-1] + " - 1\n", "        start = 0\n"),)
    decoy_note = "ignore the checkpoint and retrain from step 0 (a blanket restart)"
    return bug, alt, alt_note, decoy, decoy_note, (rank, kill_step), dict(rank=rank, step=kill_step), None, None


_BUILDERS = {"O1": _o1, "O2": _o2, "O3": _o3, "O4": _o4}


def make_mutant(seed: int, op: str, variant: str) -> Mutant:
    """Everything is drawn from private seeded streams in a fixed order, so a (seed, op, variant) is byte-stable."""
    assert SEED_RANGE[0] <= seed <= SEED_RANGE[1], f"seed {seed} outside the pilot range {SEED_RANGE}"
    assert op in OPERATORS and variant in VARIANTS
    rng = random.Random(f"mut:{seed}:{op}:{variant}")
    world = rng.choice([3, 4])
    job_seed = random.Random(f"mut-job-seed:{seed}").randrange(1, 10**6)
    decoy_warmup, decoy_shard, decoy_rank = rng.choice([5, 6, 7]), rng.choice([5, 6, 7]), rng.randrange(world)
    bug, alt, alt_note, decoy, decoy_note, kill, details, _, _ = _BUILDERS[op](variant, world, rng)
    cfg = base_config(world, job_seed, decoy_warmup, decoy_shard, decoy_rank)
    kind = ("A" if op == "O1" else "B") + f"_{op}{variant}"
    spec = JobSpec(kind, 0, cfg, bug, kill, dict(details, world=world))
    return Mutant(seed, op, variant, spec, alt, decoy, alt_note, decoy_note)


def apply_arm(job_dir: Path, mutant: Mutant, arm: str) -> None:
    """Apply an arm's edits to a materialized BUGGY job dir (gold/alt/decoy are patches over the buggy text)."""
    edits = {"gold": mutant.gold, "alt": mutant.alt, "decoy": mutant.decoy, "noop": ()}[arm]
    apply_edits(job_dir, list(edits))


def arm_edits(mutant: Mutant, arm: str) -> list:
    return list({"gold": mutant.gold, "alt": mutant.alt or (), "decoy": mutant.decoy, "noop": ()}[arm])


__all__ = ["Mutant", "make_mutant", "materialize", "apply_arm", "arm_edits", "OPERATORS", "VARIANTS", "SEED_RANGE"]
