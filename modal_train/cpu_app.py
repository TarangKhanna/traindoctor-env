"""Modal CPU container (no GPU): run the executed test suite and `prepare` on Linux. Ephemeral TD_KEY generated inside the
container per run; the owner's ~/.traindoctor key and any key/TEST material are never uploaded. Plumbing only (no secrets)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RES = "/results"
app = modal.App("traindoctor-cpu")
res_vol = modal.Volume.from_name("td-results", create_if_missing=True)

_SKIP = {".git", ".wt", ".venv", "archive", "__pycache__", ".pytest_cache", ".hypothesis", "results", "cache", "logs", ".claude"}


def _ignore(p: Path) -> bool:
    return bool(set(p.parts) & _SKIP) or p.name in (".devkey", "ledger.jsonl") or p.name.endswith((".key", ".pyc", ".pkl", ".local.json", ".local.key.txt"))


_base = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_CACHE_DIR": "1", "HF_HOME": "/opt/hf", "LANG": "C.UTF-8"})
    .pip_install("uv==0.6.6")
    .add_local_file(ROOT / "pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/build/uv.lock", copy=True)
    .run_commands(
        "cd /build && uv export --frozen --no-hashes --no-emit-project --no-header"
        " | grep -v -E '^(torch|triton|nvidia-|cuda-|mlx|inductive-mlxrl)' > req.txt"
        " && uv pip install --system --no-deps -r req.txt"
        " && pip install \"torch==$(grep -A1 '^name = \\\"torch\\\"' uv.lock | sed -n 's/version = \\\"\\(.*\\)\\\"/\\1/p')\""
        " --index-url https://download.pytorch.org/whl/cpu",
        "python -c \"from huggingface_hub import snapshot_download as s; p=s('mlx-community/Qwen3-0.6B-4bit', allow_patterns=['*.json','*.txt'], ignore_patterns=['*.safetensors*']); open('/opt/hf/REVISION','w').write(p)\"",
    )
    .env({"HF_HUB_OFFLINE": "1"})
)
image = _base.add_local_dir(ROOT, "/work", ignore=_ignore)


def _run(argv, env, timeout):
    t = time.time()
    r = subprocess.run(argv, cwd="/work", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return r.returncode, r.stdout, time.time() - t


@app.function(image=image, volumes={RES: res_vol}, cpu=4, memory=8192, timeout=3000, max_containers=1, retries=0)
def s1(run_id: str = "s1", dev: int = 48, train: int = 64, do_prepare: bool = True) -> dict:
    env = {k: v for k, v in os.environ.items()}
    env.update(TD_KEY=os.urandom(16).hex(), PYTHONPATH="/work", TD_PREPARE_OUT=f"{RES}/data", CI="true", TD_CI_UNSANDBOXED="1")  # repo-owned code; gVisor container, no sandbox-exec
    out: dict = dict(run_id=run_id, cpus=os.cpu_count())
    import torch, platform
    out["versions"] = dict(torch=str(torch.__version__), python=platform.python_version(), platform=platform.platform())
    for name, extra in (("pytest", ["-m", "not sweep"]),):
        rc, so, dt = _run(["python", "-m", "pytest", "-q", "-rfs", "-p", "no:cacheprovider", "tests/", "--ignore=tests/test_first_update.py", *extra], env, 1500)
        lines = so.splitlines()
        out[name] = dict(rc=rc, secs=dt, summary=lines[-1] if lines else "",
                         skipped=[l for l in lines if l.startswith("SKIPPED")],
                         failed=[l[:230] for l in lines if l.startswith(("FAILED", "ERROR"))], tail="\n".join(lines[-40:])[-5000:])
    # Gloo/loopback viability, independent of the suite: 2-rank all_reduce
    probe = ("import os,torch,torch.distributed as d,torch.multiprocessing as m\n"
             "def f(r):\n os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT='29533')\n d.init_process_group('gloo',rank=r,world_size=2)\n"
             " t=torch.ones(1)*(r+1);d.all_reduce(t);print('rank',r,'sum',t.item(),flush=True);d.destroy_process_group()\n"
             "if __name__=='__main__': m.spawn(f,nprocs=2)\n")
    Path("/tmp/gloo_probe.py").write_text(probe)
    rc, so, _ = _run(["python", "/tmp/gloo_probe.py"], env, 120)
    out["gloo_probe"] = dict(rc=rc, out=so[-800:])
    if do_prepare:
        rc, so, dt = _run(["python", "-m", "modal_train.prepare", "--dev", str(dev), "--train", str(train)], env, 1500)
        out["prepare"] = dict(rc=rc, secs=dt, tail=so[-1500:])
        if rc == 0:
            out["tokens"] = _tokens(Path(RES) / "data")
    d = Path(RES) / "runs"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{run_id}_s1.json").write_text(json.dumps(out, indent=1, default=str))
    res_vol.commit()
    return out


DIAG = r"""
import sys, tempfile, json
from pathlib import Path
from traindoctor_env.bundle import make_bundle, make_menu, BundleTooLarge, est_tokens, PROMPT_OVERHEAD, render, padded_config
from traindoctor_env.pilot import DEV_SEEDS, stratified_seeds
from traindoctor_env.scenario import make_scenario
from traindoctor_env.snapshot import execute, load_truth
from jobs.build import ALL_LABELS
from traindoctor_env.verifiers import certificate_ids
n = int(sys.argv[1])
with tempfile.TemporaryDirectory() as wd:
    for s in stratified_seeds(n, DEV_SEEDS):
        sc = make_scenario(s)
        snap = execute(sc, Path(wd) / str(s)); truth = load_truth(snap); menu = make_menu(snap, truth)
        cert = certificate_ids(list(snap.records), truth)
        by = {}
        for r in snap.records: by[r["kind"]] = by.get(r["kind"], 0) + 1
        try:
            b = make_bundle(snap, menu, truth); res = ("ok", b.tokens)
        except BundleTooLarge as e:
            res = ("TOO_LARGE", str(e)[-40:])
        print(json.dumps(dict(seed=s, kind=sc.kind, scen=str(sc)[:120], nrec=len(snap.records), ncert=len(cert), by=by, res=res)), flush=True)
"""


@app.function(image=image, cpu=4, memory=8192, timeout=900, max_containers=1, retries=0)
def diag(n: int = 12) -> str:
    env = dict(os.environ, TD_KEY=os.urandom(16).hex(), PYTHONPATH="/work", CI="true", TD_CI_UNSANDBOXED="1")
    Path("/tmp/diag.py").write_text(DIAG)
    return _run(["python", "/tmp/diag.py", str(n)], env, 800)[1][-6000:]


@app.local_entrypoint()
def run_diag(n: int = 12):
    print(diag.remote(n=n))


def _tokens(data: Path) -> dict:
    import pickle, statistics
    from transformers import AutoTokenizer
    from traindoctor_env.bundle import CAP_TOKENS
    from traindoctor_env.tiera import training_prompt
    tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-4bit", local_files_only=True)
    res = {}
    for name in ("dev", "train"):
        eps = pickle.loads((data / f"{name}_eps.pkl").read_bytes())  # written by this same run's prepare
        n = [len(tok.encode(training_prompt(e[1]))) for e in eps]
        res[name + "_table"] = dict(outcomes=sum(len(e[3]) for e in eps), ok=sum(o.ok for e in eps for o in e[3].values()),
                                    sandboxed=sum(o.sandboxed for e in eps for o in e[3].values()))
        res[name] = dict(n=len(n), cap=CAP_TOKENS, max=max(n), mean=statistics.mean(n), median=statistics.median(n),
                         frac_over=sum(x > CAP_TOKENS for x in n) / len(n))
    return res


@app.local_entrypoint()
def run_s1(run_id: str = "s1", do_prepare: bool = True):
    r = s1.remote(run_id=run_id, do_prepare=do_prepare)
    for k in ("pytest",):
        print(k, {x: r[k][x] for x in ("rc", "secs", "summary", "failed", "skipped")})
    print("gloo", r["gloo_probe"]); print("prepare", r.get("prepare", {}).get("rc"), r.get("prepare", {}).get("tail", "")[-300:])
    print("tokens", r.get("tokens")); print("versions", r["versions"])
    Path(f"modal_train/results/{run_id}_s1.json").write_text(json.dumps(r, indent=1, default=str))


AUDIT = r"""
import json, os, pickle, sys, tempfile, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rl.splits import TEST_SEEDS_SEALED
from traindoctor_env import reexec
from traindoctor_env.pilot import DEV_SEEDS, stratified_seeds
from traindoctor_env.scenario import make_scenario
from traindoctor_env.snapshot import execute
eps = pickle.loads(Path(sys.argv[1]).read_bytes())          # this project's own prepare output
seeds = stratified_seeds(48, DEV_SEEDS); assert not set(seeds) & set(TEST_SEEDS_SEALED)
per_kind = int(sys.argv[2]); out = Path(sys.argv[3])
chosen, seen = [], defaultdict(int)
for i, s in enumerate(seeds):
    k = make_scenario(s).kind
    if seen[k] < per_kind:
        seen[k] += 1; chosen.append(i)
KEYS = ("ok", "steps_completed", "ckpt_valid", "lost_steps", "ranks_consistent", "samples_match_reference", "params_match_reference", "fault_not_injected")
rows = []
with tempfile.TemporaryDirectory() as wd:
    for i in chosen:
        truth, bundle, menu, table = eps[i]
        snap = execute(make_scenario(seeds[i]), Path(wd) / str(seeds[i]))
        oks = [c for c in menu if table[c.id].ok]
        non = [c for c in menu if not table[c.id].ok and not table[c.id].fault_not_injected]
        pick = oks + non[:int(sys.argv[4])]
        def one(c, snap=snap, i=i, table=table):
            t0 = time.time(); cell = table[c.id]
            row = dict(ep=i, seed=seeds[i], kind=make_scenario(seeds[i]).kind, cand=c.id, role=c.role, table={k: getattr(cell, k) for k in KEYS}, table_mode=cell.exec_mode)
            try:
                o = reexec.verify_fix(snap, list(c.edits), allow_unsandboxed=False, backend="modal")
                row.update(modal={k: getattr(o, k) for k in KEYS}, modal_mode=o.exec_mode, modal_sandboxed=o.sandboxed, modal_detail=o.detail[:200])
                row["agree_ok"] = bool(o.ok) == bool(cell.ok)
                row["agree_all"] = all(row["modal"][k] == row["table"][k] for k in KEYS)
            except BaseException as e:
                row.update(error=f"{type(e).__name__}: {e}"[:300])
            row["wall_s"] = time.time() - t0
            print(json.dumps(row, default=str), flush=True); return row
        with ThreadPoolExecutor(4) as ex:
            rows += list(ex.map(one, pick))
        out.write_text(json.dumps(rows, indent=1, default=str))
"""


audit_image = _base.pip_install("modal==1.6.2").add_local_dir(ROOT, "/work", ignore=_ignore)   # the Sandbox backend needs the client inside the container


@app.function(image=audit_image, volumes={RES: res_vol}, cpu=4, memory=8192, timeout=3300, max_containers=1, retries=0)
def audit_live(per_kind: int = 1, extra_non: int = 2, data: str = "data_menu2") -> dict:
    env = dict(os.environ, TD_KEY=os.urandom(16).hex(), PYTHONPATH="/work", CI="true")   # NOT TD_CI_UNSANDBOXED: replays go to the Modal Sandbox
    Path("/tmp/audit.py").write_text(AUDIT)
    out = Path(RES) / "runs" / f"audit_live_{data}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    rc, so, dt = _run(["python", "/tmp/audit.py", f"{RES}/{data}/dev_eps.pkl", str(per_kind), str(out), str(extra_non)], env, 3200)
    res_vol.commit()
    return dict(rc=rc, secs=dt, tail=so[-3000:])


@app.local_entrypoint()
def run_audit(per_kind: int = 1, extra_non: int = 2, data: str = "data_menu2"):
    r = audit_live.remote(per_kind=per_kind, extra_non=extra_non, data=data)
    print(json.dumps(r, indent=1))


# ===================== SEALED TEST (protocol: docs/prereg/test_protocol.md; run once) =====================
BUILD_ONE = r"""
import hashlib, json, pickle, sys, tempfile
from pathlib import Path
from rl.splits import BURNED_SEEDS, TEST_SEEDS_SEALED, TRAIN_SEEDS
from traindoctor_env.bundle import make_bundle, make_menu
from traindoctor_env.pilot import DEV_SEEDS
from traindoctor_env.scenario import make_scenario
from traindoctor_env.snapshot import execute, load_truth
seed, kind, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
assert seed in TEST_SEEDS_SEALED and not ({seed} & (set(DEV_SEEDS) | set(TRAIN_SEEDS) | set(BURNED_SEEDS)))
sc = make_scenario(seed); assert sc.kind == kind
with tempfile.TemporaryDirectory() as wd:
    snap = execute(sc, Path(wd) / str(seed)); truth = load_truth(snap)
    menu = make_menu(snap, truth); b = make_bundle(snap, menu, truth)      # menu v1 (default), original prompt v1
out.write_bytes(pickle.dumps((truth, b, menu, {})))                         # no outcome table for TEST
h = lambda x: hashlib.sha256(repr(x).encode()).hexdigest()[:16]
print(json.dumps(dict(bundle=h(b.text), menu=h(menu), truth=h((truth.label, truth.fix)), records=h(b.records), tokens=b.tokens)))
"""


@app.function(image=image, volumes={RES: res_vol}, cpu=4, memory=8192, timeout=1500, max_containers=8, retries=0)
def build_test_ep(i: int, seed: int, kind: str, tag: str, key: str) -> dict:
    # ONE fixed TD_KEY for the whole TEST run: candidate/evidence ids and data salts are keyed by it (determinism needs the same key)
    env = dict(os.environ, TD_KEY=key, PYTHONPATH="/work", CI="true")
    Path("/tmp/build_one.py").write_text(BUILD_ONE)
    d = Path(RES) / f"data_test_parts_{tag}"; d.mkdir(parents=True, exist_ok=True)
    rc, so, dt = _run(["python", "/tmp/build_one.py", str(seed), kind, str(d / f"{i}.pkl")], env, 1400)
    res_vol.commit()
    last = so.strip().splitlines()[-1] if so.strip() else ""
    return dict(i=i, rc=rc, secs=dt, hashes=last if rc == 0 else None, tail="" if rc == 0 else so[-1500:])


@app.function(image=image, volumes={RES: res_vol}, cpu=1, memory=2048, timeout=600, max_containers=1, retries=0)
def assemble_test(tag: str, n: int) -> dict:
    import pickle
    import sys
    sys.path.insert(0, "/work")
    res_vol.reload()
    d = Path(RES) / f"data_test_parts_{tag}"
    eps = [pickle.loads((d / f"{i}.pkl").read_bytes()) for i in range(n)]    # our own build output
    out = Path(RES) / "data_test"; out.mkdir(parents=True, exist_ok=True)
    (out / "dev_eps.pkl").write_bytes(pickle.dumps(eps))                      # file name load_eps(.., "dev") expects
    (out / "raw_only").write_text("sealed TEST: raw completions only; scored by cpu_app.py::run_score\n")
    res_vol.commit()
    from collections import Counter
    return dict(n=len(eps), kinds=dict(Counter(e[0].kind for e in eps)), max_tokens=max(e[1].tokens for e in eps))


@app.local_entrypoint()
def run_build_test(check: bool = True):
    """Phase 1: build the 100 sealed-TEST episodes on Modal CPU (8 concurrent). check=True first builds episodes 0-2 twice
    in separate containers and requires identical hashes."""
    from modal_train import budget
    spec = json.loads((ROOT / "modal_train" / "test_seeds.local.json").read_text())
    seeds, kinds = spec["seeds"], spec["kinds"]
    key = (ROOT / "modal_train" / "test.local.key.txt").read_text().strip()   # untracked, excluded from the image; passed as an argument
    assert len(seeds) == 100 and len(set(seeds)) == 100
    cpu_s = lambda secs: secs * 4   # ledger models CPU in cpu-seconds at cpu_per_hour (as audit_live)  # noqa: E731
    budget.project("cpu_per_hour", cpu_s(100 * 180))
    budget.record("test_build2", "test_build", "cpu_per_hour", cpu_s(100 * 180), "projected")
    total = 0.0
    if check:
        a = list(build_test_ep.map(range(3), seeds[:3], kinds[:3], ["chkA"] * 3, [key] * 3))
        b = list(build_test_ep.map(range(3), seeds[:3], kinds[:3], ["chkB"] * 3, [key] * 3))
        for x, y in zip(a, b):
            assert x["rc"] == 0 and y["rc"] == 0, (x["tail"], y["tail"])
            assert x["hashes"] == y["hashes"], f"NONDETERMINISTIC build ep {x['i']}: {x['hashes']} vs {y['hashes']}"
            total += x["secs"] + y["secs"]
        print("determinism check PASS (3 episodes built twice in separate containers):", [x["hashes"] for x in a], flush=True)
    rs = list(build_test_ep.map(range(100), seeds, kinds, ["main"] * 100, [key] * 100))
    bad = [r for r in rs if r["rc"] != 0]
    total += sum(r["secs"] for r in rs)
    budget.record("test_build2", "test_build", "cpu_per_hour", cpu_s(total), "actual")
    assert not bad, [b["tail"] for b in bad[:2]]
    print(assemble_test.remote("main", 100), f"build container secs {total:.0f}", flush=True)


# ---- sealed TEST phase 3: live Modal-Sandbox replay of unique (episode, chosen repair) pairs, then scoring ----
POLICIES = ["base", "sft", "csft"] + [f"v0_s{s}" for s in range(6)] + [f"v2_s{s}" for s in range(6)]
G_TEST, ROUND = 4, 20

REPLAY_ONE = r"""
import dataclasses, json, pickle, sys, tempfile, time
from pathlib import Path
from traindoctor_env import reexec
from traindoctor_env.scenario import make_scenario
from traindoctor_env.snapshot import execute
ep, repairs, out = int(sys.argv[1]), json.loads(sys.argv[2]), Path(sys.argv[3])
eps = pickle.loads(Path(sys.argv[4]).read_bytes())          # this project's own build output
truth, bundle, menu, _ = eps[ep]
rows = []
with tempfile.TemporaryDirectory() as wd:
    snap = execute(make_scenario(truth.seed), Path(wd) / str(truth.seed))
    same = {r["id"] for r in bundle.records} <= {r["id"] for r in snap.records}
    for rid in repairs:
        t0 = time.time(); row = dict(ep=ep, repair=rid, role=next(c.role for c in menu if c.id == rid))
        try:
            if not same:
                raise RuntimeError("rebuilt snapshot does not contain the bundle's record ids")
            c = next(c for c in menu if c.id == rid)
            o = reexec.verify_fix(snap, list(c.edits), allow_unsandboxed=False, backend="modal")
            row["outcome"] = dataclasses.asdict(o)
        except BaseException as e:
            row["error"] = f"{type(e).__name__}: {e}"[:400]
        row["wall_s"] = time.time() - t0
        print(json.dumps(dict(ep=ep, repair=rid, ok=row.get("outcome", {}).get("ok"), mode=row.get("outcome", {}).get("exec_mode"), err=row.get("error"), s=round(row["wall_s"], 1)), default=str), flush=True)
        rows.append(row)
out.write_text(json.dumps(rows, default=str))
"""


@app.function(image=audit_image, volumes={RES: res_vol}, cpu=4, memory=8192, timeout=2400, max_containers=8, retries=0)
def replay_ep(ep: int, repairs: list, key: str) -> dict:
    res_vol.reload()
    env = dict(os.environ, TD_KEY=key, PYTHONPATH="/work", CI="true")   # NOT TD_CI_UNSANDBOXED: Modal Sandbox only
    Path("/tmp/replay_one.py").write_text(REPLAY_ONE)
    rc, so, dt = _run(["python", "/tmp/replay_one.py", str(ep), json.dumps(repairs), "/tmp/rows.json", f"{RES}/data_test/dev_eps.pkl"], env, 2300)
    rows = json.loads(Path("/tmp/rows.json").read_text()) if rc == 0 and Path("/tmp/rows.json").exists() else []
    return dict(ep=ep, rc=rc, secs=dt, rows=rows, tail="" if rc == 0 else so[-1500:])


def _load_rollouts(policy: str) -> list:
    return json.loads((Path(RES) / "runs" / f"test_{policy}_eval.json").read_text())["completions"]


@app.function(image=image, volumes={RES: res_vol}, cpu=1, memory=4096, timeout=900, max_containers=1, retries=0)
def collect_pairs() -> dict:
    """Unique (episode, repair id) pairs chosen by valid answers across the 15 policies' TEST rollouts."""
    import pickle
    import sys
    sys.path.insert(0, "/work")
    from traindoctor_env.tiera import parse_answer
    res_vol.reload()
    eps = pickle.loads((Path(RES) / "data_test" / "dev_eps.pkl").read_bytes())
    pairs, n_valid, n = set(), 0, 0
    for p in POLICIES:
        comps = _load_rollouts(p)
        assert len(comps) == G_TEST * len(eps), (p, len(comps))
        for i, c in enumerate(comps):
            n += 1
            a = parse_answer(c["text"], {m.id for m in eps[i // G_TEST][2]})
            if a:
                n_valid += 1
                pairs.add((i // G_TEST, a.repair))
    return dict(pairs=sorted(pairs), rollouts=n, valid=n_valid)


@app.function(image=image, volumes={RES: res_vol}, cpu=2, memory=8192, timeout=900, max_containers=1, retries=0)
def score_test(rounds: int, replay_rows: list) -> dict:
    """Per-rollout V0/V1/V2 for episodes in the fully replayed rounds. V2 only from live modal_sandbox outcomes."""
    import pickle
    import sys
    sys.path.insert(0, "/work")
    from traindoctor_env.reexec import FixOutcome
    from traindoctor_env.tiera import parse_answer
    from traindoctor_env.bundle import answer_verdict
    res_vol.reload()
    eps = pickle.loads((Path(RES) / "data_test" / "dev_eps.pkl").read_bytes())
    out_by = {}
    for r in replay_rows:
        o = r.get("outcome")
        if o and o.get("exec_mode") == "modal_sandbox" and o.get("sandboxed") is True:
            o = dict(o); o["tampered"] = tuple(o.get("tampered") or ())
            out_by[(r["ep"], r["repair"])] = FixOutcome(**o)
        else:
            out_by[(r["ep"], r["repair"])] = None     # replay crash/timeout/non-modal mode: unverified-failure
    dummy = FixOutcome(ok=False, bit_identical=False, params_match_reference=False, samples_match_reference=False,
                       restores_consistent=False, no_duplicate_updates=False, lost_steps=0, ranks_consistent=False, sandboxed=False)
    n_eps = ROUND * rounds
    rolls = []
    for p in POLICIES:
        for i, c in enumerate(_load_rollouts(p)):
            e = i // G_TEST
            if e >= n_eps:
                continue
            truth, bundle, menu, _ = eps[e]
            a = parse_answer(c["text"], {m.id for m in menu})
            if a is None:
                rolls.append(dict(policy=p, ep=e, kind=truth.kind, valid=False, v0=False, v1=False, v2=False, claimed=False, replay="none", unsafe=False, repair=None))
                continue
            o = out_by[(e, a.repair)]                  # KeyError = a chosen pair was never replayed: fail loudly
            v = answer_verdict(truth, bundle, menu, {a.repair: o or dummy}, a.label, list(a.evidence), a.repair, a.claims_fixed)
            rolls.append(dict(policy=p, ep=e, kind=truth.kind, valid=True, v0=bool(v.v0), v1=bool(v.v1), v2=bool(v.v2 and o is not None),
                              claimed=bool(a.claims_fixed), replay="ok" if o is not None else "error",
                              unsafe=bool(next(m for m in menu if m.id == a.repair).destructive), repair=a.repair))
    pairs = [dict(ep=r["ep"], repair=r["repair"], role=r.get("role"), error=r.get("error"), wall_s=r.get("wall_s"),
                  **({k: r["outcome"].get(k) for k in ("ok", "exec_mode", "sandboxed", "steps_completed", "ckpt_valid", "lost_steps", "ranks_consistent",
                                                       "samples_match_reference", "params_match_reference", "fault_not_injected", "sample_gaps", "sample_dups", "path")} if r.get("outcome") else {}))
             for r in replay_rows]
    return dict(rollouts=rolls, pairs=pairs, rounds=rounds, episodes=n_eps)


@app.local_entrypoint()
def run_replay():
    """Phase 3: collect unique pairs -> replay in rounds (8 concurrent sandboxes) with the ledger gate -> score. Single shot."""
    from modal_train import budget
    cpu_s = lambda secs: secs * 4   # noqa: E731
    key = (ROOT / "modal_train" / "test.local.key.txt").read_text().strip()
    col = collect_pairs.remote()
    pairs = [tuple(x) for x in col["pairs"]]
    print(f"rollouts {col['rollouts']}, valid {col['valid']}, unique pairs {len(pairs)} (worst-case est ${len(pairs) * 0.02:.2f})", flush=True)
    R = Path("modal_train/results"); R.mkdir(exist_ok=True)
    (R / "test_pairs.json").write_text(json.dumps(col))
    all_rows, done, per_pair = [], 0, 0.02
    t_start_spent = budget.spent_usd()
    for r in range(5):
        mine = [p for p in pairs if p[0] // ROUND == r]
        proj = len(mine) * max(0.02, per_pair)
        if budget.spent_usd() + proj > budget.CAP_USD:
            print(f"STOP before round {r}: spent ${budget.spent_usd():.2f} + projected ${proj:.2f} > cap ${budget.CAP_USD:.2f}", flush=True)
            break
        by = {}
        for e, rep in mine:
            by.setdefault(e, []).append(rep)
        rid = f"test_replay_r{r}"
        budget.record(rid, "test_replay", "cpu_per_hour", cpu_s(len(mine) * 60), "projected")
        res = list(replay_ep.map(list(by), [by[e] for e in by], [key] * len(by)))
        secs = sum(x["secs"] for x in res)
        budget.record(rid, "test_replay", "cpu_per_hour", cpu_s(secs + sum(w.get("wall_s", 0) for x in res for w in x["rows"])), "actual")
        rows = [w for x in res for w in x["rows"]]
        crashed = [x for x in res if x["rc"] != 0]
        for x in crashed:   # a crashed episode container: its pairs are errors (not retried)
            rows += [dict(ep=x["ep"], repair=rep, error=f"container rc={x['rc']}: {x['tail'][-200:]}") for rep in by[x["ep"]]]
        got = {(w["ep"], w["repair"]) for w in rows}
        rows += [dict(ep=e, repair=rep, error="no result returned") for e, rep in mine if (e, rep) not in got]
        all_rows += rows; done = r + 1
        per_pair = (budget.spent_usd() - t_start_spent) / max(1, len([p for p in pairs if p[0] // ROUND < done]))
        (R / f"test_replay_r{r}.json").write_text(json.dumps(rows, default=str))
        print(f"round {r}: {len(mine)} pairs, {sum(1 for w in rows if w.get('error'))} errors, {sum(1 for w in rows if (w.get('outcome') or {}).get('ok'))} recovered, observed ${per_pair:.4f}/pair so far", flush=True)
    sc = score_test.remote(done, all_rows)
    (R / "test_scored.json").write_text(json.dumps(sc, indent=1, default=str))
    print(f"scored {len(sc['rollouts'])} rollouts over {sc['episodes']} episodes ({done} rounds)", flush=True)


@app.local_entrypoint()
def run_assemble():
    print(assemble_test.remote("main", 100))
