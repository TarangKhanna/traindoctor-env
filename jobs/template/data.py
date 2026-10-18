"""Training data position, and shard loading."""
from __future__ import annotations

import json
import os
import time

import torch

IN_DIM, OUT_DIM = 8, 4


class Sampler:
    """The data position (epoch, index) of this rank. Batches come from the cluster's data service, which owns the
    shuffle and the samples: next_batch() asks it for the batch at the current position and advances; an epoch
    ends after steps_per_epoch batches."""

    def __init__(self, num_samples: int, world_size: int, rank: int, batch_size: int, seed: int):
        self.steps_per_epoch = num_samples // world_size // batch_size
        self.epoch, self.index = 0, 0

    def next_batch(self):
        import td_client  # the data service client, provided by the cluster runtime

        _, x, y = td_client.fetch(self.epoch, self.index)
        self.index += 1
        if self.index == self.steps_per_epoch:
            self.epoch, self.index = self.epoch + 1, 0
        return x, y

    def state(self) -> torch.Tensor:
        return torch.tensor([self.epoch, self.index])

    def load_state(self, t: torch.Tensor) -> None:
        self.epoch, self.index = int(t[0]), int(t[1])


def shard_path(cfg: dict, rank: int, k: int) -> str:
    return os.path.join(cfg["data"]["root"], cfg["data"]["shard_dirs"][str(rank)], f"shard{k}.idx")


def open_shard(cfg: dict, rank: int, k: int, log) -> dict:
    """Shards are published by the data service; wait until this rank's shard shows up."""
    path = shard_path(cfg, rank, k)
    if not os.path.exists(path):
        log(f"waiting for shard {k} at {path}")
        while not os.path.exists(path):
            time.sleep(0.2)
    with open(path) as f:
        meta = json.load(f)
    log(f"loaded shard {k} from {path}")
    return meta
