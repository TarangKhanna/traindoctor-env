"""LOCAL, CPU-only: real re-execution of TRAIN/DEV episodes -> pickled (truth, bundle, menu, table) + oracle SFT jsonl.
Never touches TEST seeds. Output (gitignored) goes to modal_train/cache/ and is uploaded by `modal run app.py::upload`.
The pickles are written and read only by this project's own code."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Optional untracked file modal_train/prepare_opts.json {"menu": "v2", "dev": 48, "train": 0}: lets the fixed command
# (`prepare --dev 48 --train 64`, a fixed spec) build another menu version / split sizes. Absent -> unchanged behaviour.
OPTS = json.loads((HERE / "prepare_opts.json").read_text()) if (HERE / "prepare_opts.json").exists() else {}
MENU = OPTS.get("menu", "v1")
CACHE = Path(os.environ.get("TD_PREPARE_OUT") or HERE / ("cache" if MENU == "v1" else f"cache_menu{MENU[1:]}"))


def main():
    from jobs.build import KINDS
    from rl.splits import TEST_SEEDS_SEALED, TRAIN_SEEDS
    from traindoctor_env.bundle import make_bundle, make_menu, outcome_table
    from traindoctor_env.oracle import reference_solution
    from traindoctor_env.pilot import DEV_SEEDS, stratified_seeds
    from traindoctor_env.scenario import make_scenario
    from traindoctor_env.snapshot import execute, load_truth

    p = argparse.ArgumentParser()
    p.add_argument("--dev", type=int, default=48)
    p.add_argument("--train", type=int, default=64, help="train episodes for GRPO prompts (per-kind round robin)")
    a = p.parse_args()
    a.dev, a.train = OPTS.get("dev", a.dev), OPTS.get("train", a.train)
    CACHE.mkdir(exist_ok=True)
    from collections import Counter
    from traindoctor_env import reexec
    reexec.DEV_CONTAINER_MODE = reexec.ci_unsandboxed_allowed()   # Linux + CI + TD_CI_UNSANDBOXED only

    def train_seeds(n):
        by = {k: [] for k in KINDS}
        for s in TRAIN_SEEDS:
            k = make_scenario(s).kind
            if len(by[k]) < n // len(KINDS) + 1:
                by[k].append(s)
        return [by[KINDS[i % len(KINDS)]][i // len(KINDS)] for i in range(n)]

    for name, seeds in (("dev", stratified_seeds(a.dev, DEV_SEEDS)), ("train", train_seeds(a.train) if a.train else [])):
        if not seeds:
            continue
        assert not set(seeds) & set(TEST_SEEDS_SEALED)
        eps, sft = [], []
        with tempfile.TemporaryDirectory() as wd:
            for s in seeds:
                snap = execute(make_scenario(s), Path(wd) / str(s)); truth = load_truth(snap)
                menu = make_menu(snap, truth, MENU); b = make_bundle(snap, menu, truth)
                eps.append((truth, b, menu, outcome_table(snap, menu)))
                lab, ev, _ = reference_solution(snap, truth)
                rep = next(c.id for c in menu if c.role == "correct")
                sft.append(dict(seed=s, kind=make_scenario(s).kind, target=json.dumps(dict(label=lab, evidence=ev, repair=rep, claims_fixed=True))))
                print(name, s, flush=True)
        (CACHE / f"{name}_eps.pkl").write_bytes(pickle.dumps(eps))
        (CACHE / f"{name}_oracle.jsonl").write_text("\n".join(json.dumps(x) for x in sft))
        outs = [o for _, _, _, t in eps for o in t.values()]
        modes = Counter(o.exec_mode for o in outs)
        stats = dict(candidates=len(outs), ok=sum(o.ok for o in outs), fault_not_injected=sum(o.fault_not_injected for o in outs),
                     by_exec_mode=dict(modes), by_exec_mode_ok={m: sum(o.ok for o in outs if o.exec_mode == m) for m in modes},
                     dev_container_mode=reexec.DEV_CONTAINER_MODE)
        (CACHE / f"{name}_exec_mode.json").write_text(json.dumps(stats, indent=1, sort_keys=True))
        if MENU != "v1":
            (CACHE / "menu_version").write_text(MENU + "\n")
            (CACHE / "prompt_version").write_text("v2\n")      # menu v2 is always paired with prompt v2
        print("wrote", name, len(eps), json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
