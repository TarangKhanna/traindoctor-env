"""Environment-sourced oracle gold answers for TRAIN episodes (oracle.reference_solution + the menu's 'correct' repair).
No LLM output. Writes out/sft/train.jsonl {seed, kind, prompt, target}. Failing seeds are logged and skipped."""
from __future__ import annotations

import argparse, json, sys, tempfile, time
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--per-kind", type=int, default=60); p.add_argument("--out", default="out/sft")
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jobs.build import KINDS
    from rl.splits import TRAIN_SEEDS
    from traindoctor_env.bundle import make_bundle, make_menu
    from traindoctor_env.oracle import reference_solution
    from traindoctor_env.scenario import make_scenario
    from traindoctor_env.snapshot import execute, load_truth
    from traindoctor_env.tiera import training_prompt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for s in TRAIN_SEEDS:
        by[make_scenario(s).kind].append(s)
    got, skipped, t0 = defaultdict(int), [], time.time()
    with tempfile.TemporaryDirectory() as wd, open(out / "train.jsonl", "w") as f:
        for i in range(a.per_kind):
            for k in KINDS:
                if i >= len(by[k]):
                    continue
                s = by[k][i]
                try:
                    snap = execute(make_scenario(s), Path(wd) / str(s)); truth = load_truth(snap)
                    menu = make_menu(snap, truth); b = make_bundle(snap, menu, truth)
                    lab, ev, _ = reference_solution(snap, truth)
                    rep = next(c.id for c in menu if c.role == "correct")
                    tgt = json.dumps(dict(label=lab, evidence=ev, repair=rep, claims_fixed=True))
                    f.write(json.dumps(dict(seed=s, kind=k, prompt=training_prompt(b), target=tgt, tokens=b.tokens)) + "\n"); f.flush()
                    got[k] += 1
                except Exception as e:  # noqa: BLE001
                    skipped.append((s, k, repr(e)[:100])); print("SKIP", skipped[-1], flush=True)
            print(i, dict(got), f"{time.time()-t0:.0f}s", flush=True)
    (out / "skipped.json").write_text(json.dumps(skipped))


if __name__ == "__main__":
    main()
