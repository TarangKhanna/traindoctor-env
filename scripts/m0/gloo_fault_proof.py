"""M0 §5.4-1: 4-rank Gloo COLLECTIVE_MISMATCH proof (CPU, macOS).
Usage: python scripts/m0/gloo_fault_proof.py --mode fault|fixed --runs N --out DIR
fault: at step FAULT_STEP rank 2 takes a rank-dependent branch (broadcast instead of all_reduce).
fixed: config flag rank_branch=False removes the branch; job runs STEPS steps and checkpoints.
"""
import argparse, hashlib, json, os, shutil, socket, sys, time, traceback
from datetime import timedelta
import torch, torch.distributed as dist, torch.multiprocessing as mp

WORLD, STEPS, FAULT_STEP, FAULT_RANK, TIMEOUT_S, WATCHDOG_S = 4, 20, 3, 2, 8, 60


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


class Tracer:
    def __init__(self, path, rank): self.f, self.rank, self.seq = open(path, "w"), rank, 0
    def call(self, step, op, fn, tensor):
        rec = dict(rank=self.rank, step=step, op=op, seq=self.seq, shape=list(tensor.shape), t_enter=time.time())
        self.seq += 1
        try:
            fn(); rec["t_exit"] = time.time()
        except BaseException as e:
            rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"; rec["t_error"] = time.time(); raise
        finally:
            self.f.write(json.dumps(rec) + "\n"); self.f.flush()


def checksum(model):
    h = hashlib.sha256()
    for p in model.state_dict().values(): h.update(repr(p.flatten().tolist()).encode())
    return h.hexdigest()[:16]


def worker(rank, cfg):
    out = cfg["out"]; res = dict(rank=rank, last_step=-1, status="running", t_start=time.time())
    def save():
        json.dump(res, open(f"{out}/result_rank{rank}.json", "w"))
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(cfg["port"]),
                      TORCH_FR_BUFFER_SIZE="2000")
    tr = Tracer(f"{out}/trace_rank{rank}.jsonl", rank)
    torch.manual_seed(cfg["seed"])
    try:
        dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=TIMEOUT_S))
        model = torch.nn.Linear(8, 4); opt = torch.optim.SGD(model.parameters(), lr=0.1)
        g = torch.Generator().manual_seed(1000 + rank)
        for step in range(STEPS):
            x = torch.randn(16, 8, generator=g)
            model.zero_grad(); model(x).pow(2).mean().backward()
            for name, p in model.named_parameters():
                t = p.grad
                if cfg["rank_branch"] and rank == FAULT_RANK and step == FAULT_STEP:
                    tr.call(step, "broadcast", lambda: dist.broadcast(t, src=0), t)   # divergent collective
                else:
                    tr.call(step, "all_reduce", lambda: dist.all_reduce(t), t); t /= WORLD
            opt.step(); res["last_step"] = step; save()
        # sync weights check + checkpoint (rank 0 only)
        res["checksum"] = checksum(model)
        if rank == 0:
            torch.save(model.state_dict(), f"{out}/ckpt.pt")
            json.dump(dict(step=STEPS, world=WORLD, shape={k: list(v.shape) for k, v in model.state_dict().items()}),
                      open(f"{out}/ckpt_manifest.json", "w"))
        dist.barrier()
        if rank == 0:
            m2 = torch.nn.Linear(8, 4); m2.load_state_dict(torch.load(f"{out}/ckpt.pt")); res["reload_ok"] = checksum(m2) == res["checksum"]
        res["status"] = "ok"
    except BaseException as e:
        res.update(status="error", exc_type=type(e).__name__, exc_first_line=str(e).splitlines()[0][:200] if str(e) else "")
        try:
            import torch._C._distributed_c10d as c; open(f"{out}/fr_rank{rank}.json", "wb").write(c._dump_fr_trace_json(True, False)); res["fr_dump"] = "written"
        except Exception as e2: res["fr_dump"] = f"unavailable: {type(e2).__name__}: {str(e2)[:120]}"
        raise
    finally:
        res["t_end"] = time.time(); res["elapsed"] = res["t_end"] - res["t_start"]; save()


def run_once(cfg):
    os.makedirs(cfg["out"], exist_ok=True); cfg["port"] = free_port(); t0 = time.time()
    ctx = mp.start_processes(worker, args=(cfg,), nprocs=WORLD, join=False, start_method="spawn")
    killed = False
    while any(p.is_alive() for p in ctx.processes):   # poll ourselves: ctx.join() would SIGTERM siblings of the first failure
        time.sleep(0.5)
        if time.time() - t0 > WATCHDOG_S:
            killed = True
            for p in ctx.processes: p.kill()
            break
    for p in ctx.processes: p.join(5)
    ranks = []
    for r in range(WORLD):
        try: d = json.load(open(f"{cfg['out']}/result_rank{r}.json"))
        except Exception: d = dict(rank=r, status="no_result")
        d["exitcode"] = ctx.processes[r].exitcode; ranks.append(d)
    return dict(wall=time.time() - t0, watchdog_killed=killed, ranks=ranks)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["fault", "fixed"], required=True)
    ap.add_argument("--runs", type=int, default=1); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); summaries = []
    for i in range(a.runs):
        d = f"{a.out}/{a.mode}_run{i}"; shutil.rmtree(d, ignore_errors=True)
        s = run_once(dict(out=d, seed=a.seed, rank_branch=(a.mode == "fault"))); summaries.append(s)
        sig = [(r["status"], r.get("exc_type"), r.get("exc_first_line"), r.get("last_step"), r["exitcode"]) for r in s["ranks"]]
        s["signature"] = sig
        print(f"run{i} wall={s['wall']:.1f}s killed={s['watchdog_killed']}")
        for r in s["ranks"]:
            print(f"  rank{r['rank']} {r['status']} exit={r['exitcode']} last_step={r.get('last_step')} t={r.get('elapsed', 0):.1f}s "
                  f"{r.get('exc_type','')} {r.get('exc_first_line','')} csum={r.get('checksum','')} reload={r.get('reload_ok','')}")
    json.dump(summaries, open(f"{a.out}/{a.mode}_summary.json", "w"), indent=1)
    if a.mode == "fault":
        print("DETERMINISTIC(error types/last_step/exit codes):", len({json.dumps([(x[0], x[1], x[3], x[4]) for x in s["signature"]]) for s in summaries}) == 1)
        print("DETERMINISTIC(full messages):", len({json.dumps(s["signature"]) for s in summaries}) == 1)
    else:
        ok = all(all(r["status"] == "ok" for r in s["ranks"]) and len({r["checksum"] for r in s["ranks"]}) == 1 and s["ranks"][0].get("reload_ok") for s in summaries)
        print("FIXED_PASS:", ok)


if __name__ == "__main__":
    main()
