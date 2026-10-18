"""Checkpoints: ckpt/<id>/state.pt (model, optimizer momentum, sampler position) + ckpt/MANIFEST.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

MANIFEST = Path("ckpt", "MANIFEST.json")


def ckpt_id(step: int) -> str:
    return f"ck{step:04d}"


def digest(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def commit(cid: str, step: int, world_size: int) -> None:
    """Record the checkpoint in the manifest; entries at or after this step are superseded."""
    state = torch.load(Path("ckpt", cid, "state.pt"))
    entry = dict(ckpt_id=cid, step=step, world_size=world_size,
                 shards=[dict(file=f"{cid}/state.pt", sha256=digest(state), shape=list(state["model.fc1.weight"].shape))])
    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    entries = [e for e in entries if e["step"] < step] + [entry]
    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def save(step: int, model, opt, sampler, rank: int, world_size: int, group) -> str:
    cid = ckpt_id(step)
    if rank == 0:
        Path("ckpt", cid).mkdir(parents=True, exist_ok=True)
        state = {f"model.{k}": v.detach().clone() for k, v in model.state_dict().items()}
        state.update({f"opt.{i}": opt.state[p]["momentum_buffer"].detach().clone()
                      for i, p in enumerate(model.parameters())})
        state.update(sampler=sampler.state(), step=torch.tensor([step]))
        torch.save(state, Path("ckpt", cid, "state.pt"))
        commit(cid, step, world_size)
    dist.barrier(group=group)
    return cid


def restore(cid: str, model, opt, sampler, rank: int) -> int:
    state = torch.load(Path("ckpt", cid, "state.pt"))
    model.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")})
    sd = opt.state_dict()
    sd["state"] = {int(k[4:]): {"momentum_buffer": v.clone()} for k, v in state.items() if k.startswith("opt.")}
    opt.load_state_dict(sd)
    if "sampler" in state:  # checkpoints from before the sampler was tracked have no position
        sampler.load_state(state["sampler"])
    return int(state["step"])
