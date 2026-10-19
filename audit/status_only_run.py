"""Run the status-only baselines on new development seeds 80000-80999 (post-hoc diagnostic audit, not part of the
original experiment, not the consumed held-out set). The true class comes from the seed's scenario and is used only for scoring. The snapshot truth is loaded only because the environment's bundle builder takes it; the classifiers receive the bundle text alone."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PER_CLASS = 40
LO, HI = 80000, 81000
FORBIDDEN = [range(10000, 10400), range(20000, 20400), range(7001, 7009), range(5000, 5800), range(7000, 7200),
             range(40000, 40100), range(60000, 70000), range(70000, 71010), (101, 201, 202)]


def cls_of(sc) -> str | None:
    if sc.kind == "A2":
        return "A2"
    if sc.kind == "A1":
        return "A1_src" if sc.variant == 0 else "A1_cfg"
    return None


def pick_seeds():
    from traindoctor_env.scenario import make_scenario
    got: dict[str, list[int]] = {"A1_src": [], "A1_cfg": [], "A2": []}
    for s in range(LO, HI):
        c = cls_of(make_scenario(s))
        if c and len(got[c]) < PER_CLASS:
            got[c].append(s)
        if all(len(v) == PER_CLASS for v in got.values()):
            break
    seeds = sorted(x for v in got.values() for x in v)
    assert all(len(v) == PER_CLASS for v in got.values()), {k: len(v) for k, v in got.items()}
    for s in seeds:
        assert LO <= s < HI and not any(s in r for r in FORBIDDEN), s
    return got


def one(args):
    seed, truth_cls, work = args
    os.environ.setdefault("TD_KEY", "status-audit-" + "0" * 16)
    from audit.status_only_baseline import classify_job, classify_process
    from traindoctor_env.bundle import make_bundle, make_menu
    from traindoctor_env.scenario import make_scenario
    from traindoctor_env.snapshot import execute, load_truth
    snap = execute(make_scenario(seed), Path(work) / f"s{seed}")
    t = load_truth(snap)
    text = make_bundle(snap, make_menu(snap, t), t).text
    pp, pj = classify_process(text), classify_job(text)       # bundle text only
    status = [ln for ln in text.splitlines() if " rank " in ln and (" failed exit=" in ln or " killed exit=" in ln)]
    return dict(seed=seed, truth=truth_cls, pred_process=pp, pred_job=pj, bundle_sha256=hashlib.sha256(text.encode()).hexdigest(),
                process_lines=status)


def main(run_id: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    got = pick_seeds()
    out = Path("results/status_audit") / run_id
    out.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(prefix="status_audit_")
    jobs = [(s, c, work) for c, v in got.items() for s in v]
    (out / "seeds.json").write_text(json.dumps(got, indent=1))
    with ThreadPoolExecutor(4) as ex, open(out / "raw.jsonl", "w") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, sort_keys=True) + "\n")
            f.flush()


if __name__ == "__main__":
    main(sys.argv[1])
