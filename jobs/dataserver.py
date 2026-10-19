"""The trusted data server, owned by the launcher (it lives in the launcher's process; ranks reach it through
a private socketpair each, so who is asking is decided by the socket, never by the message).

Samples are generated here from (job seed, per-scenario salt, sample id). The salt is derived from a key that
only the owner's side holds and is never put in a job's environment, files or arguments, so job code cannot
produce the training data itself; it can only ask for the batch at a position (epoch, index) of ITS rank.

The log of what was served is EVIDENCE (served is not consumed: a job may prefetch). Verdicts do not rest on
it alone; see traindoctor_env/reexec.py."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

IN_DIM, OUT_DIM = 8, 4
MAX_LINE = 4096          # a request is a few dozen bytes; anything larger is abuse
MAX_EPOCH = 1000
MAX_SERVED = 10_000      # per rank per run
KEY_FILE = Path.home() / ".traindoctor" / "key"


def _key() -> bytes:
    if "TD_KEY" in os.environ:
        return os.environ["TD_KEY"].encode()
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, os.urandom(32).hex().encode())
        os.close(fd)
    return KEY_FILE.read_bytes()


def salt_for(scenario_seed: int | str) -> bytes:
    """Per-scenario secret salt: stable for a scenario (so a snapshot is reproducible and a reference run uses
    the same data as a verification run) and unknowable without the key."""
    return hashlib.sha256(_key() + b"|salt|" + str(scenario_seed).encode()).digest()


@dataclass(frozen=True)
class SaltSchedule:
    """Which secret salt keys the sample VALUES at each data position (epoch * steps_per_epoch + index). Positions
    below `boundary` use `base` (data that existing checkpoints legitimately already consumed); positions from
    `boundary` on use `fresh`, a per-verification nonce the harness draws and never stores. Checkpoints or sample
    records produced in an earlier run therefore cannot be valid for the data a verification run is served."""
    base: bytes
    fresh: bytes
    boundary: int

    def at(self, position: int) -> bytes:
        return self.base if position < self.boundary else self.fresh

    @classmethod
    def fresh_for(cls, base: bytes, boundary: int) -> "SaltSchedule":
        return cls(base, hashlib.sha256(base + b"|nonce|" + os.urandom(32)).digest(), boundary)


def salt_at(salt: "bytes | SaltSchedule", position: int) -> bytes:
    return salt.at(position) if isinstance(salt, SaltSchedule) else salt


@dataclass(frozen=True)
class DataSpec:
    world: int
    num_samples: int
    batch_size: int
    seed: int

    @property
    def steps_per_epoch(self) -> int:
        return self.num_samples // self.world // self.batch_size


def schedule_ids(spec: DataSpec, rank: int, epoch: int, index: int) -> list[int]:
    """Rank r owns the ids with id % world == r, shuffled per epoch by a generator seeded with (seed, epoch,
    rank); the batch at (epoch, index) is a slice of that order."""
    shard = list(range(rank, spec.num_samples, spec.world))
    g = torch.Generator().manual_seed(spec.seed * 1000003 + epoch * 1009 + rank)
    order = [shard[i] for i in torch.randperm(len(shard), generator=g).tolist()]
    return order[index * spec.batch_size:(index + 1) * spec.batch_size]


def sample_x(salt: bytes, job_seed: int, i: int) -> torch.Tensor:
    h = hashlib.sha256(salt + b"|x|" + f"{job_seed}:{i}".encode()).digest()
    return torch.randn(IN_DIM, generator=torch.Generator().manual_seed(int.from_bytes(h[:8], "big")))


def batch_tensors(salt: bytes, job_seed: int, ids: list[int]):
    xs = torch.stack([sample_x(salt, job_seed, i) for i in ids])
    return xs, xs[:, :OUT_DIM] * 0.5 + 0.1


def _enc(t: torch.Tensor) -> str:
    return base64.b64encode(t.contiguous().numpy().tobytes()).decode()


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class DataServer:
    """handle(rank, line) -> response dict, or None when no response is due. Never raises on bad input."""

    def __init__(self, spec: DataSpec, salt: bytes):
        self.spec, self.salt = spec, salt
        self.served: dict[int, list[tuple]] = {r: [] for r in range(spec.world)}   # (epoch, index, ids, acked_at_serve)
        self.acked: dict[int, int] = {r: 0 for r in range(spec.world)}
        self.digests: dict[int, str | None] = {r: None for r in range(spec.world)}
        self.closed = False
        self.activity = 0

    def close(self) -> None:
        self.closed = True

    def parse(self, rank: int, line: bytes) -> tuple[str, dict]:
        """('batch', req) | ('step', msg) | ('error', {error}) | ('drop', {}); drop = cut the connection."""
        if len(line) > MAX_LINE:
            return "drop", {}
        try:
            msg = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            return "error", dict(error="malformed request")
        if not isinstance(msg, dict):
            return "error", dict(error="malformed request")
        op = msg.get("op")
        if op == "batch":
            if set(msg) != {"op", "epoch", "index"}:  # e.g. a claimed rank or step: who asks is the socket, not the text
                return "error", dict(error="unexpected fields")
            if not (_is_int(msg["epoch"]) and _is_int(msg["index"])):
                return "error", dict(error="epoch and index must be integers")
            if not (0 <= msg["epoch"] < MAX_EPOCH and 0 <= msg["index"] < self.spec.steps_per_epoch):
                return "error", dict(error="position out of range")
            return "batch", msg
        if op == "step":
            return "step", msg
        return "error", dict(error="unknown op")

    def note_step(self, rank: int, msg: dict) -> None:
        """Evidence only. Acks must arrive in order (n = previous + 1); anything else is ignored."""
        self.activity += 1
        d = msg.get("digest")
        if msg.get("n") == self.acked[rank] + 1 and set(msg) <= {"op", "n", "digest"}:
            self.acked[rank] += 1
            self.digests[rank] = d if isinstance(d, str) and len(d) <= 64 else None

    def serve(self, rank: int, req: dict) -> dict:
        self.activity += 1
        if self.closed:
            return dict(error="run closed")
        if len(self.served[rank]) >= MAX_SERVED:
            return dict(error="too many requests")
        ids = schedule_ids(self.spec, rank, req["epoch"], req["index"])
        x, y = batch_tensors(salt_at(self.salt, req["epoch"] * self.spec.steps_per_epoch + req["index"]), self.spec.seed, ids)
        self.served[rank].append((req["epoch"], req["index"], ids, self.acked[rank]))
        return dict(ids=ids, x=_enc(x), x_shape=list(x.shape), y=_enc(y), y_shape=list(y.shape))
