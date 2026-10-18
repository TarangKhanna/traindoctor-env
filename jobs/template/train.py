"""Data-parallel training of a small MLP on CPU with Gloo (one process per rank).

Per run (runs/<RUN_ID>/): logs/rank{r}.jsonl and trace/rank{r}.jsonl (collectives). Launched from the job
directory by the cluster's launcher with RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, RUN_ID and RESUME_FROM in
the environment.
"""
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

import ckpt
import config
import data

N_BUCKETS = 2


class Jsonl:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def write(self, **rec) -> None:
        self.f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.f.flush()


class Comm:
    """Collectives with a per-rank trace: every call is recorded on entry and on exit."""

    def __init__(self, trace: Jsonl, world_size: int, group):
        self.trace, self.world, self.group, self.seq = trace, world_size, group, 0

    def _run(self, op: str, t: torch.Tensor, step: int, bucket: int, fn) -> None:
        self.trace.write(ev="enter", seq=self.seq, step=step, op=op, bucket=bucket, numel=t.numel())
        fn()
        self.trace.write(ev="exit", seq=self.seq)
        self.seq += 1

    def all_reduce(self, t, step, bucket):
        def fn():
            dist.all_reduce(t, group=self.group)
            t.div_(self.world)
        self._run("all_reduce", t, step, bucket, fn)

    def reduce(self, t, step, bucket):
        self._run("reduce", t, step, bucket, lambda: dist.reduce(t, dst=0, group=self.group))

    def broadcast(self, t, step, bucket):
        self._run("broadcast", t, step, bucket, lambda: dist.broadcast(t, src=0, group=self.group))


class MLP(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(data.IN_DIM, hidden)
        self.fc2 = torch.nn.Linear(hidden, data.OUT_DIM)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def sync_grads(comm: Comm, flat: torch.Tensor, step: int, rank: int, cfg: dict) -> None:
    """Average the flattened gradients across ranks, one bucket at a time."""
    mode = cfg["sync"]["mode"]
    if step >= cfg["sync"]["warmup_steps"]:
        mode = cfg["sync"]["per_rank"].get(str(rank), mode)
    for b, bucket in enumerate(flat.chunk(N_BUCKETS)):
        if mode == "reduce_bcast" and b == N_BUCKETS - 1:
            comm.reduce(bucket, step, b)
            comm.broadcast(bucket, step, b)
            bucket.div_(comm.world)
        else:
            comm.all_reduce(bucket, step, b)


def main() -> None:
    torch.set_num_threads(1)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    cfg = config.load("job/config.yaml")
    run = Path("runs", os.environ["RUN_ID"])
    log_f = Jsonl(run / "logs" / f"rank{rank}.jsonl")
    status = dict(step=-1)

    def log(msg: str, level: str = "INFO") -> None:
        log_f.write(step=status["step"], level=level, msg=msg)

    log(f"start world_size={world} resume_from={os.environ.get('RESUME_FROM') or 'none'}")
    torch.manual_seed(cfg["seed"])
    model = MLP(cfg["hidden"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    sampler = data.Sampler(cfg["data"]["num_samples"], world, rank, cfg["batch_size"], cfg["seed"])
    start = 0
    if os.environ.get("RESUME_FROM"):
        start = ckpt.restore(os.environ["RESUME_FROM"], model, opt, sampler, rank)
        status["step"] = start - 1
        log(f"resumed from {os.environ['RESUME_FROM']} at step {start}")
    os.environ.update(MASTER_ADDR="127.0.0.1")
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=60))  # ranks start at different times
    group = dist.new_group(timeout=timedelta(seconds=cfg["pg_timeout_s"]))  # collectives time out quickly
    comm = Comm(Jsonl(run / "trace" / f"rank{rank}.jsonl"), world, group)
    for step in range(start, cfg["steps"]):
        if step % cfg["data"]["shard_steps"] == 0:
            data.open_shard(cfg, rank, step // cfg["data"]["shard_steps"], log)
        x, y = sampler.next_batch()
        Path("progress").mkdir(exist_ok=True)
        Path("progress", f"rank{rank}.json").write_text(json.dumps(dict(step=step, sampler=sampler.state().tolist())))
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        params = list(model.parameters())
        flat = torch.cat([p.grad.flatten() for p in params])
        flat = torch.cat([flat, torch.zeros(-flat.numel() % N_BUCKETS)])
        sync_grads(comm, flat, step, rank, cfg)
        off = 0
        for p in params:
            p.grad.copy_(flat[off:off + p.numel()].view_as(p))
            off += p.numel()
        opt.step()
        status["step"] = step
        log(f"step {step} loss={loss.item():.3f}")
        if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == cfg["steps"]:
            log(f"checkpoint {ckpt.save(step + 1, model, opt, sampler, rank, world, group)} committed")
        time.sleep(cfg["step_time_s"])
    log("training complete")


if __name__ == "__main__":
    main()
