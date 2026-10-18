"""Modal entry points. Imported by `modal run`; the core env never imports this. Nothing here runs at import time except
object definitions (no compute). GPU: env TD_GPU (default A100-80GB; H100 opt-in). No web endpoints, no always-on services."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import modal

GPU = os.environ.get("TD_GPU", "A100-80GB")
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR, RES_DIR, DATA_DIR = "/cache", "/results", "/results/data"
# Worst-case seconds charged by the budget guard before launch (== function timeout).
TIMEOUT = dict(smoke=1800, synth=900, sft=3600, grpo=5400, eval=3600)

def _exclude(p: Path) -> bool:
    """True = leave out. Keeps only .py files; drops caches, tests and anything under a data/cache/results dir."""
    bad = {"__pycache__", "tests", "cache", "results", "logs", "out", "data"}
    if set(p.parts) & bad:
        return True
    return False if p.is_dir() else p.suffix != ".py"


app = modal.App("traindoctor-train")
hf_vol = modal.Volume.from_name("td-hf-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("td-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.1", "trl==1.16.0", "peft==0.21.2", "accelerate==1.15.0", "datasets", "pydantic==2.14.1")
    .env({"HF_HOME": f"{CACHE_DIR}/hf", "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false"})
    # Python source of only these packages: no data dirs, no .env, no tests/, so no sealed TEST data or secrets ride along.
    .add_local_python_source("traindoctor_env", "jobs", "rl", "modal_train", ignore=_exclude)
)
common = dict(image=image, gpu=GPU, volumes={CACHE_DIR: hf_vol, RES_DIR: res_vol}, max_containers=1, retries=0)


def _finish(name: str, res: dict, t0: float, run_id: str) -> dict:
    res.update(run_id=run_id, fn=name, container_s=time.time() - t0, finished=time.time())
    from modal_train.trainlib import gpu_info
    res["gpu_info"] = gpu_info()
    d = Path(RES_DIR) / "runs"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{run_id}_{name}.json").write_text(json.dumps(res, indent=1, default=str))
    res_vol.commit(); hf_vol.commit()
    return res


@app.function(timeout=TIMEOUT["smoke"], **common)
def smoke(model: str = "Qwen/Qwen3-0.6B", run_id: str = "smoke") -> dict:
    from modal_train import trainlib as T
    t0 = time.time()
    out = f"{RES_DIR}/smoke_{run_id}"
    a = T.run_sft(model, DATA_DIR, out, steps=3, limit=16, batch=2, accum=1)
    g = T.run_grpo(model, DATA_DIR, out, "V2", sft_adapter=f"{out}/adapter", steps=1, group=4, max_new=256, limit=4, prompts_per_step=1)
    return _finish("smoke", dict(sft=a, grpo=g), t0, run_id)


@app.function(timeout=TIMEOUT["synth"], **common)
def synth_smoke(model: str = "Qwen/Qwen3-0.6B", run_id: str = "synth") -> dict:
    """PLUMBING ONLY: 2 SFT steps + 1 GRPO update on synthetic_smoke data (modal_train/synth.py). Not real episodes."""
    from datasets import Dataset
    from modal_train import synth, trainlib as T
    import torch
    t0 = time.time()
    out = f"/tmp/synth_{run_id}"
    tok = T.load_tok(model)
    rows = synth.rows(16)
    sft_ds = Dataset.from_list([dict(prompt=r["prompt"], completion=r["completion"]) for r in rows])
    ntok = sum(len(tok(r["prompt"] + r["completion"])["input_ids"]) for r in rows[:8])
    a = T.run_sft(model, "", out, steps=2, batch=4, accum=1, ds=sft_ds)
    a["tokens_processed"] = ntok  # 2 steps x batch 4 = the first 8 rows
    a["tok_s"] = ntok / a["wall_s"]
    gds = Dataset.from_list([dict(prompt=r["prompt"], idx=r["idx"]) for r in rows])
    torch.cuda.reset_peak_memory_stats()
    g = T.run_grpo(model, "", out, "stub", sft_adapter=f"{out}/adapter", steps=1, group=4, max_new=128, prompts_per_step=1,
                   ds=gds, reward_fn=synth.stub_reward)
    return _finish("synth_smoke", dict(dataset=synth.DATASET, versions=_versions(), sft=a, grpo=g), t0, run_id)


def _versions() -> dict:
    import importlib.metadata as m
    return {p: m.version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate")}


@app.function(timeout=TIMEOUT["sft"], **common)
def sft(model: str, run_id: str, epochs: int = 2, lr: float = 1e-4, batch: int = 4, accum: int = 4, save_at: int = 0, steps: int = -1, init_run: str = "", data: str = "data") -> dict:
    from modal_train import trainlib as T
    t0 = time.time()
    ia = f"{RES_DIR}/{init_run}/adapter" if init_run else None
    return _finish("sft", T.run_sft(model, f"{RES_DIR}/{data}", f"{RES_DIR}/{run_id}", steps=steps, epochs=epochs, lr=lr, batch=batch, accum=accum, save_at=save_at, init_adapter=ia), t0, run_id)


@app.function(timeout=TIMEOUT["grpo"], **common)
def grpo(model: str, run_id: str, arm: str, sft_run: str, steps: int = 100, group: int = 8, lr: float = 5e-6, adapter_name: str = "adapter", seed: int = 0, data: str = "data") -> dict:
    from modal_train import trainlib as T
    t0 = time.time()
    r = T.run_grpo(model, f"{RES_DIR}/{data}", f"{RES_DIR}/{run_id}", arm, sft_adapter=f"{RES_DIR}/{sft_run}/{adapter_name}", steps=steps, group=group, lr=lr, seed=seed)
    r["seed"] = seed
    return _finish("grpo", r, t0, run_id)


@app.function(timeout=TIMEOUT["eval"], **common)
def eval_(model: str, run_id: str, adapter_run: str = "", g: int = 4, limit: int = 0, batch: int = 8, adapter_name: str = "adapter", variant: str = "none", data: str = "data") -> dict:
    from modal_train import trainlib as T
    t0 = time.time()
    ad = f"{RES_DIR}/{adapter_run}/{adapter_name}" if adapter_run else None
    return _finish("eval", T.run_eval(model, f"{RES_DIR}/{data}", ad, g=g, limit=limit, batch=batch, variant=variant), t0, run_id)


def _launch(fn_name: str, remote, **kw):
    """Budget-guarded launch: project worst case, run, then replace the projection with the actual GPU-seconds."""
    from modal_train import budget
    rid = kw["run_id"] if "run_id" in kw else uuid.uuid4().hex[:8]
    proj = budget.project(GPU, TIMEOUT[fn_name])
    budget.record(rid, fn_name, GPU, TIMEOUT[fn_name], "projected")
    print(f"[budget] projected total ${proj:.2f} (cap ${budget.CAP_USD})", flush=True)
    res = remote.remote(**kw)
    budget.record(rid, fn_name, GPU, res["container_s"], "actual")
    print(json.dumps({k: v for k, v in res.items() if k not in ("completions",)}, indent=1, default=str)[:6000])
    return res


@app.local_entrypoint()
def upload(cache: str = "cache", data: str = "data"):
    """Upload locally prepared episodes (modal_train/<cache>, e.g. cache or cache_v2prompt) to volume dir /<data>. No compute."""
    src = Path(__file__).resolve().parent / cache
    with res_vol.batch_upload(force=True) as b:
        for f in sorted(src.glob("*")):
            b.put_file(str(f), f"/{data}/{f.name}")
            print("upload", f.name)


@app.local_entrypoint()
def run_synth(model: str = "Qwen/Qwen3-0.6B"):
    _launch("synth", synth_smoke, model=model, run_id=f"synth-{uuid.uuid4().hex[:6]}")


@app.local_entrypoint()
def run_smoke(model: str = "Qwen/Qwen3-0.6B"):
    _launch("smoke", smoke, model=model, run_id=f"smoke-{uuid.uuid4().hex[:6]}")


@app.local_entrypoint()
def run_sft(run_id: str, model: str = "Qwen/Qwen3-4B", epochs: int = 2, lr: float = 1e-4, batch: int = 4, accum: int = 4, save_at: int = 0, steps: int = -1, init_run: str = "", data: str = "data"):
    _launch("sft", sft, data=data, model=model, run_id=run_id, epochs=epochs, lr=lr, batch=batch, accum=accum, save_at=save_at, steps=steps, init_run=init_run)


@app.local_entrypoint()
def run_grpo(run_id: str, arm: str, sft_run: str, model: str = "Qwen/Qwen3-4B", steps: int = 100, lr: float = 5e-6, adapter_name: str = "adapter", seed: int = 0, data: str = "data"):
    assert arm in ("V0", "V2")
    _launch("grpo", grpo, data=data, model=model, run_id=run_id, arm=arm, sft_run=sft_run, steps=steps, lr=lr, adapter_name=adapter_name, seed=seed)


@app.local_entrypoint()
def run_eval(run_id: str, model: str = "Qwen/Qwen3-4B", adapter_run: str = "", g: int = 4, limit: int = 0, batch: int = 8, adapter_name: str = "adapter", variant: str = "none", data: str = "data"):
    _launch("eval", eval_, data=data, model=model, run_id=run_id, adapter_run=adapter_run, g=g, limit=limit, batch=batch, adapter_name=adapter_name, variant=variant)


@app.local_entrypoint()
def download(dest: str = "modal_train/results"):
    """Pull run JSONs (no adapters) from the results volume. Read-only, no compute."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    for e in res_vol.listdir("/runs"):
        data = b"".join(res_vol.read_file(e.path))
        (Path(dest) / Path(e.path).name).write_bytes(data)
        print("got", e.path)
