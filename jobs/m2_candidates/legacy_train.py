"""Gloo data-parallel training worker on a toy MLP (CPU).

Per run (job_dir/runs/<run_id>/): logs/rank{r}.jsonl, trace/rank{r}.jsonl, result_rank{r}.json.
Gradients are flattened and synced in two equal buckets; a rank override may make one rank
enter a different collective (the executable COLLECTIVE_MISMATCH mechanism).
"""
from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from jobs import manifest
from jobs.config import JobConfig

IN_DIM, OUT_DIM, N_BUCKETS = 8, 4, 2
TIMEOUT_MARKERS = ("Timed out", "Connection closed by peer", "Connection reset", "Broken pipe")


class ShapeError(Exception):
    pass


class CkptNotFound(Exception):
    pass


def classify(e: BaseException, cfg: JobConfig) -> tuple[str, str]:
    """Map an exception to a canonical (class, message). Peers of a hung collective may see
    either a timeout or a closed connection; both are the same event."""
    if isinstance(e, manifest.ManifestError):
        return "MANIFEST_INVALID", str(e)
    if isinstance(e, ShapeError):
        return "SHAPE_ERROR", str(e)
    if isinstance(e, CkptNotFound):
        return "CKPT_NOT_FOUND", str(e)
    if isinstance(e, RuntimeError) and any(m in str(e) for m in TIMEOUT_MARKERS):
        return "COLLECTIVE_TIMEOUT", f"collective timed out (timeout={cfg.pg_timeout_s:g}s)"
    first = (str(e).splitlines() or [""])[0][:200]
    return "UNKNOWN", f"{type(e).__name__}: {first}"


class Jsonl:
    def __init__(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(p, "a")

    def write(self, **rec) -> None:
        self.f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.f.flush()


class MLP(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(IN_DIM, hidden)
        self.fc2 = torch.nn.Linear(hidden, OUT_DIM)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def batch(cfg: JobConfig, rank: int, step: int):
    g = torch.Generator().manual_seed(cfg.seed * 100003 + rank * 1009 + step)
    x = torch.randn(16, IN_DIM, generator=g)
    return x, x[:, :OUT_DIM] * 0.5 + 0.1


def ckpt_id(step: int) -> str:
    return f"ck{step:04d}"


def opt_buffers(model: MLP, opt) -> dict:
    return {f"opt.{i}": opt.state[p]["momentum_buffer"].detach().clone()
            for i, p in enumerate(model.parameters()) if p in opt.state}


def full_digest(model: MLP, opt, step: int) -> str:
    """Checksum of everything that determines the future of training: params, momentum, global step
    (data position and RNG are pure functions of (seed, rank, step))."""
    return manifest.digest({**model.state_dict(), **opt_buffers(model, opt), "step": torch.tensor([step])})


def load_ckpt(cfg: JobConfig, job_dir: Path, rank: int, model: MLP, opt) -> int:
    entries = manifest.load(job_dir)
    manifest.validate(job_dir, entries)
    entry = next((e for e in entries if e["ckpt_id"] == cfg.resume_from), None)
    if entry is None:
        raise CkptNotFound(f"checkpoint {cfg.resume_from} not found in manifest")
    shards = entry["shards"]
    split = entry["shard_layout"] == "split"
    if split:
        want = [cfg.hidden // cfg.world_size, IN_DIM]
        if rank >= len(shards):
            raise ShapeError(f"rank {rank}: no checkpoint shard (checkpoint has {len(shards)} shards, "
                             f"world_size={cfg.world_size})")
        if shards[rank]["shape"] != want:
            raise ShapeError(f"rank {rank}: checkpoint shard shape {shards[rank]['shape']} "
                             f"does not match expected {want}")
    parts = [torch.load(Path(job_dir) / "ckpt" / s["file"]) for s in shards]
    state = dict(parts[0])
    if split:
        for k in ("fc1.weight", "fc1.bias"):
            state[k] = torch.cat([p[k] for p in parts])
    bufs = {k: state.pop(k) for k in [k for k in state if k.startswith("opt.")]}
    model.load_state_dict(state)
    sd = opt.state_dict()
    sd["state"] = {int(k[4:]): {"momentum_buffer": v.clone()} for k, v in bufs.items()}
    opt.load_state_dict(sd)
    return entry["step"]


def commit_ckpt(cfg: JobConfig, job_dir: Path, rank: int, model: MLP, opt, done: int) -> str:
    cid = ckpt_id(done)
    cdir = Path(job_dir) / "ckpt" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if cfg.shard_layout == "split":
        rows = cfg.hidden // cfg.world_size
        shard = {k: sd[k][rank * rows:(rank + 1) * rows].clone() for k in ("fc1.weight", "fc1.bias")}
        if rank == 0:
            shard.update({k: sd[k] for k in ("fc2.weight", "fc2.bias")}, **opt_buffers(model, opt))
        torch.save(shard, cdir / f"shard{rank}.pt")
    elif rank == 0:
        torch.save({**sd, **opt_buffers(model, opt)}, cdir / "shard0.pt")
    dist.barrier()
    if rank == 0:
        manifest.append(job_dir, manifest.make_entry(job_dir, cid, done, cfg.world_size,
                                                     cfg.shard_layout, cfg.ckpt_format_version))
    dist.barrier()
    return cid


def pick_op(ov: dict, step: int, bucket: int) -> str:
    sync = ov.get("grad_sync")
    if step >= ov.get("grad_sync_from_step", 0):
        if sync == "broadcast" and bucket == 0:
            return "broadcast"
        if sync == "reduce" and bucket == N_BUCKETS - 1:
            return "reduce"
    return "all_reduce"


def run_collective(op: str, t: torch.Tensor, world: int) -> None:
    if op == "all_reduce":
        dist.all_reduce(t)
        t /= world
    elif op == "broadcast":
        dist.broadcast(t, src=0)
    else:
        dist.reduce(t, dst=0)


def worker(rank: int, cfg_dict: dict, job_dir: str, run_id: str, port: int, preempt_after: int | None) -> None:
    torch.set_num_threads(1)
    cfg = JobConfig(**cfg_dict)
    job_dir = Path(job_dir)
    run = job_dir / "runs" / run_id
    log, tr = Jsonl(run / "logs" / f"rank{rank}.jsonl"), Jsonl(run / "trace" / f"rank{rank}.jsonl")
    ov = cfg.rank_overrides.get(str(rank), {})
    res = dict(rank=rank, status="error", last_step=-1, exc_class=None, exc_msg=None, digest=None)
    try:
        log.write(step=-1, level="INFO", msg=f"start world_size={cfg.world_size} resume_from={cfg.resume_from} "
                                             f"overrides={json.dumps(ov, sort_keys=True)}")
        if ov.get("profile"):
            log.write(step=-1, level="WARN", msg="profiler enabled, expect step-time overhead")
        if ov.get("dataloader_prefetch", 2) != 2:
            log.write(step=-1, level="INFO", msg=f"dataloader prefetch={ov['dataloader_prefetch']}")
        torch.manual_seed(cfg.seed)
        model = MLP(cfg.hidden)
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
        start = 0
        if cfg.resume_from:
            start = load_ckpt(cfg, job_dir, rank, model, opt)
            log.write(step=start - 1, level="INFO", msg=f"resumed from {cfg.resume_from} at step {start}")
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        dist.init_process_group("gloo", rank=rank, world_size=cfg.world_size,
                                timeout=timedelta(seconds=cfg.pg_timeout_s))
        cseq = 0
        for step in range(start, cfg.steps):
            x, y = batch(cfg, rank, step)
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            params = list(model.parameters())
            flat = torch.cat([p.grad.flatten() for p in params])
            if flat.numel() % N_BUCKETS:
                flat = torch.cat([flat, torch.zeros(N_BUCKETS - flat.numel() % N_BUCKETS)])
            for b, bucket in enumerate(flat.chunk(N_BUCKETS)):
                op = pick_op(ov, step, b)
                tr.write(ev="enter", cseq=cseq, step=step, op=op, bucket=b, numel=bucket.numel())
                run_collective(op, bucket, cfg.world_size)
                tr.write(ev="exit", cseq=cseq)
                cseq += 1
            off = 0
            for p in params:
                p.grad.copy_(flat[off:off + p.numel()].view_as(p))
                off += p.numel()
            opt.step()
            res["last_step"] = step
            log.write(step=step, level="INFO", msg=f"step {step} loss={loss.item():.3f}")
            done = step + 1
            if done % cfg.ckpt_every == 0 or done == cfg.steps:
                cid = commit_ckpt(cfg, job_dir, rank, model, opt, done)
                log.write(step=step, level="INFO", msg=f"checkpoint {cid} committed")
            if preempt_after == done:
                (run / f"preempt_ready{rank}").write_text("1")
                time.sleep(3600)
        log.write(step=cfg.steps - 1, level="INFO", msg="training complete")
        res["digest"] = full_digest(model, opt, cfg.steps)
        res["status"] = "ok"
    except BaseException as e:  # noqa: BLE001 - every failure must be recorded, then exit nonzero
        res["exc_class"], res["exc_msg"] = classify(e, cfg)
        log.write(step=res["last_step"], level="ERROR", msg=f"{res['exc_class']}: {res['exc_msg']}")
    (run / f"result_rank{rank}.json").write_text(json.dumps(res))
    os._exit(0 if res["status"] == "ok" else 1)
