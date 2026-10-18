"""Training data position, and shard loading (scenario version 2: shards are published by a background producer)."""
from __future__ import annotations

import json
import os
import threading
import time

import torch

IN_DIM, OUT_DIM = 8, 4


class Sampler:
    """The data position (epoch, index) of this rank. Batches come from the cluster's data service, which owns the
    shuffle and the samples. Which batch a step asks for is read from the step's shard index (`positions`), which
    the shard producer publishes; an epoch ends after steps_per_epoch batches."""

    def __init__(self, num_samples: int, world_size: int, rank: int, batch_size: int, seed: int):
        self.steps_per_epoch = num_samples // world_size // batch_size
        self.epoch, self.index = 0, 0
        self.shard_k, self.shard_steps, self.positions = 0, 1, []

    def set_shard(self, k: int, shard_steps: int, meta: dict) -> None:
        self.shard_k, self.shard_steps, self.positions = k, shard_steps, meta.get("positions", [])

    def next_batch(self):
        import td_client  # the data service client, provided by the cluster runtime

        step = self.epoch * self.steps_per_epoch + self.index
        epoch, index = self.positions[step - self.shard_k * self.shard_steps]
        _, x, y = td_client.fetch(epoch, index)
        self.index += 1
        if self.index == self.steps_per_epoch:
            self.epoch, self.index = self.epoch + 1, 0
        return x, y

    def state(self) -> torch.Tensor:
        return torch.tensor([self.epoch, self.index])

    def load_state(self, t: torch.Tensor) -> None:
        self.epoch, self.index = int(t[0]), int(t[1])


def shard_dir(cfg: dict, rank: int) -> str:
    return os.path.join(cfg["data"]["root"], cfg["data"]["shard_dirs"][str(rank)])


def shard_path(cfg: dict, rank: int, k: int) -> str:
    return os.path.join(shard_dir(cfg, rank), f"shard{k}.idx")


def ready_path(cfg: dict, rank: int, k: int) -> str:
    return shard_path(cfg, rank, k) + ".ready"


def _write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _produce(cfg: dict, rank: int, n_shards: int) -> None:
    d, prod, steps = cfg["data"], cfg["data"]["producer"], cfg["data"]["shard_steps"]
    spe = d["num_samples"] // cfg["world_size"] // cfg["batch_size"]
    for k in range(n_shards):
        positions = [list(divmod(s, spe)) for s in range(k * steps, (k + 1) * steps)]
        if k >= prod["cached_shards"]:     # shards beyond the staged ones are fetched at this rank's rate
            time.sleep(prod["shard_mb"] / prod["throttle_mb_s"][str(rank)])
        _write(shard_path(cfg, rank, k), {"shard": k, "positions": positions})
        _write(ready_path(cfg, rank, k), {"shard": k})


def start_producer(cfg: dict, rank: int) -> None:
    """Publish this rank's shards in the background: index file first, ready marker last."""
    os.makedirs(shard_dir(cfg, rank), exist_ok=True)
    n_shards, steps = -(-cfg["steps"] // cfg["data"]["shard_steps"]), cfg["data"]["shard_steps"]
    for k in range(n_shards):
        if os.path.exists(ready_path(cfg, rank, k)):   # left by an earlier attempt
            os.remove(ready_path(cfg, rank, k))
        # index files are preallocated with placeholder positions; the producer fills them in as shards are fetched
        _write(shard_path(cfg, rank, k), {"shard": k, "positions": [[0, 0]] * steps})
    threading.Thread(target=_produce, args=(cfg, rank, n_shards), daemon=True).start()


def open_shard(cfg: dict, rank: int, k: int, log) -> dict:
    """Wait until the producer has marked this rank's shard ready, then read its index."""
    path = ready_path(cfg, rank, k)
    if not os.path.exists(path):
        log(f"waiting for shard {k} at {shard_path(cfg, rank, k)}")
        while not os.path.exists(path):
            time.sleep(0.05)
    with open(shard_path(cfg, rank, k)) as f:
        meta = json.load(f)
    log(f"loaded shard {k} from {shard_path(cfg, rank, k)}")
    return meta
