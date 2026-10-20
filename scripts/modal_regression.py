"""Small Modal gVisor regression suite for the CURRENT code (one seed per kind, 4 cases each).

  TD_KEY=00ff PYTHONPATH=. python scripts/modal_regression.py [--cap-usd 0.5] [--list]

For A1_src, A1_cfg, A2, B_SKIP, B_DUP: gold must be ACCEPTED; null and two incorrect repairs (cases from
scripts/redteam_suite.py) must be REJECTED. Every verdict comes from reexec.Replay(backend='modal'); nothing here touches
the verifier. Exit 0 only if every expectation holds and every row ran in modal_sandbox. An `error` row is a violation.
Writes regression_results/modal_regression.{json,md} (or $TD_RESULTS_DIR)."""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from traindoctor_env.scenario import make_scenario  # noqa: E402

# one seed per kind, all in 90000-90999 (checked disjoint from earlier seed lists in tests/test_modal_regression_cases.py)
SEEDS = {"A1_src": 90005, "A1_cfg": 90014, "A2": 90003, "B_SKIP": 90001, "B_DUP": 90000}
KIND_OF = {"A1_src": ("A1", 0), "A1_cfg": ("A1", 1), "A2": ("A2", 0), "B_SKIP": ("B_SKIP", 0), "B_DUP": ("B_DUP", 0)}
# cases per kind: id -> must be accepted. F11 = world_size:1 shortcut; F03 = identity all_reduce (A only);
# F16 = restart from scratch (restore returns 0); F09 = zeroed model saved into the checkpoint.
CASES = {
    "A1_src": {"gold": True, "null": False, "F11": False, "F03": False},
    "A1_cfg": {"gold": True, "null": False, "F11": False, "F03": False},
    "A2": {"gold": True, "null": False, "F11": False, "F03": False},
    "B_SKIP": {"gold": True, "null": False, "F16": False, "F09": False},
    "B_DUP": {"gold": True, "null": False, "F16": False, "F09": False},
}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-usd", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    for name, (kind, var) in KIND_OF.items():
        sc = make_scenario(SEEDS[name])
        assert (sc.kind, sc.variant) == (kind, var), f"{name}: seed {SEEDS[name]} builds {sc.kind}/{sc.variant}"
    if a.list:
        for n, c in CASES.items():
            print(n, SEEDS[n], c)
        return 0
    if not os.environ.get("TD_KEY"):
        sys.exit("TD_KEY (any hex) must be set")
    import importlib.util
    if importlib.util.find_spec("modal") is None:
        site = os.environ.get("TD_MODAL_SITE")
        if not site:
            sys.exit("modal client not importable (set TD_MODAL_SITE)")
        sys.path.append(site)
    import redteam_suite as rs
    from concurrent.futures import ThreadPoolExecutor
    from traindoctor_env import modal_backend as sb
    from traindoctor_env import reexec
    from traindoctor_env.oracle import reference_solution
    from traindoctor_env.snapshot import execute, load_truth
    assert not reexec.DEV_CONTAINER_MODE and not reexec.ALLOW_UNSANDBOXED
    rs.FORMAT_CHECK = "v1"
    led = rs.Ledger(a.cap_usd)
    rs._install_capture()
    sb._IMAGE = sb.image()
    jobs = []
    with tempfile.TemporaryDirectory() as wd:
        for name, want in CASES.items():
            sc = make_scenario(SEEDS[name])
            snap = execute(sc, Path(wd) / name)
            truth = load_truth(snap)
            _, _, gold = reference_solution(snap, truth)
            cases = {c.id: c for c in rs.build_cases(sc.kind, sc.variant, snap.files, truth.details, gold)}
            ce = snap.reference["config"]["ckpt_every"]
            for cid in want:
                jobs.append((snap, cases[cid], name, ce))
        first = rs.judge(*jobs[0], led)
        rows = [first]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            rows += [f.result() for f in [ex.submit(rs.judge, *j, led) for j in jobs[1:]]]
    order = {(n, c): i for i, (n, cs) in enumerate(CASES.items()) for c in cs}
    rows.sort(key=lambda r: [i for (n, c), i in order.items() if n == r["task"] and c == r["id"]][0])
    viol = []
    L = ["| kind | seed | case | expected | got | first rejecting invariant | exec_mode | ok |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        want = CASES[r["task"]][r["id"]]
        got = "ERROR" if r.get("error") else ("accepted" if r["accepted"] else "rejected")
        exp = "accepted" if want else "rejected"
        bad = []
        if r.get("error"):
            bad.append("error: " + r["error"])
        elif r["accepted"] != want:
            bad.append(f"expected {exp}, got {got}")
        if r.get("exec_mode") != "modal_sandbox":
            bad.append(f"exec_mode={r.get('exec_mode')}")
        inv = "-" if r["accepted"] else (r.get("invariant") or "-")
        L.append(f"| {r['task']} | {SEEDS[r['task']]} | {r['id']} | {exp} | {got} | {inv} | {r.get('exec_mode')} | {'yes' if not bad else 'NO'} |")
        viol += [f"{r['task']}/{r['id']}: {b}" for b in bad]
    try:
        modal_v = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        import modal
        modal_v = getattr(modal, "__version__", "unknown")
    meta = dict(sha=_git_sha(), date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                python=platform.python_version(), modal=modal_v, spent_usd=round(led.spent(), 4), violations=viol)
    L += ["", f"commit {meta['sha']}  date {meta['date']}  python {meta['python']}  modal {meta['modal']}  "
              f"spend (script counter) ${meta['spent_usd']:.3f}", f"violations: {len(viol)}"] + [f"- {v}" for v in viol]
    out = Path(os.environ.get("TD_RESULTS_DIR") or ROOT / "regression_results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "modal_regression.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, sort_keys=True, default=str))
    (out / "modal_regression.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
