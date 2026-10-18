"""indep2: build menu-only data for many seeds WITHOUT executing any job (materialize the buggy job dir, read its files,
build Truth exactly as snapshot.execute does, call make_menu). Output: pickle of per-bundle dicts."""
import json, os, pickle, sys, tempfile, shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from jobs.build import materialize
from jobs.config import job_config
from traindoctor_env.scenario import Scenario, Truth, job_spec, make_scenario
from traindoctor_env.snapshot import job_files
from traindoctor_env.bundle import make_menu, padded_config

def one(sc):
    spec = job_spec(sc)
    d = Path(tempfile.mkdtemp())
    try:
        materialize(d, spec.config, spec.bug)
        files, cfg = job_files(d), job_config(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    truth = Truth(sc.seed, sc.kind, sc.variant, sc.label, spec.task, [asdict(e) for e in spec.fix], spec.details)
    snap = SimpleNamespace(seed=sc.seed, files=files, config=cfg)
    menu = make_menu(snap, truth)
    return dict(seed=sc.seed, kind=sc.kind, variant=sc.variant, label=sc.label, world=sc.world_size,
                config_text=padded_config(files["job/config.yaml"]),
                items=[dict(id=c.id, text=c.diff(files), role=c.role, destructive=c.destructive) for c in menu])

if __name__ == "__main__":
    out = sys.argv[1]; lo, hi, mode = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rows = []
    for s in range(lo, hi):
        base = make_scenario(s)
        if mode == "natural":
            scs = [base]
        else:   # counterfactual group: same seed/world/job_seed, every kind
            scs = [Scenario(s, k, v, base.world_size, base.job_seed) for k, v in
                   [("A1", 0), ("A1", 1), ("A2", 0), ("B_SKIP", 0), ("B_DUP", 0), ("B_OK", 0)]]
        for sc in scs:
            try:
                r = one(sc); r["group"] = s; rows.append(r)
            except Exception as e:
                print("fail", s, sc.kind, repr(e)[:120], file=sys.stderr)
    pickle.dump(rows, open(out, "wb"))
    print(len(rows), "bundles")
