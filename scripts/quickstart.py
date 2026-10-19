"""TrainDoctor quickstart: CPU/Gloo only, no paid API, no Modal needed.

  PYTHONPATH=. python scripts/quickstart.py [--seed 123] [--out DIR] [--fake F11] [--modal]

1. Build a failed multi-rank job from a seed and run it (real 4-process Gloo job: one rank enters a different collective).
2. Show what an agent sees (public records only), then the gold repair (oracle).
3. Re-run the repaired job locally. This run is NOT sandboxed: the verifier marks it INVALID and refuses to count it,
   even though every recovery invariant is computed and passes. That refusal is the point.
4. --modal: the verified path. Replays the SAME scenario (same --seed) with the same gold repair and the same fake repair
   (--fake, default F11 = world_size:1) in a disposable Modal Sandbox (gVisor, no network) via scripts/redteam_suite.py;
   needs a Modal account and an importable modal client (TD_MODAL_SITE is an optional override), about $0.03.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("TD_KEY", os.urandom(16).hex())   # keys the data server's salt; any random hex

from traindoctor_env import reexec  # noqa: E402
from traindoctor_env.oracle import reference_solution  # noqa: E402
from traindoctor_env.scenario import make_scenario  # noqa: E402
from traindoctor_env.snapshot import execute, load_truth  # noqa: E402

INVARIANTS = ("params_match_reference", "samples_match_reference", "restores_consistent", "no_duplicate_updates",
              "ranks_consistent", "ckpt_valid", "states_match_reference")


def show_diff(edits, files) -> None:
    for e in edits:
        a = files[e.path]
        b = a.replace(e.old, e.new, 1)
        sys.stdout.writelines(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{e.path}", f"b/{e.path}", n=3))


def report(title: str, o) -> None:
    mode = o.exec_mode if o.sandboxed else "local_unsandboxed"
    print(f"  [{title}] exec_mode={mode} sandboxed={o.sandboxed} ok={o.ok} steps_completed={o.steps_completed} "
          f"lost_steps={o.lost_steps} sample_gaps={o.sample_gaps} sample_dups={o.sample_dups}")
    print("   invariants: " + ", ".join(f"{k}={getattr(o, k)}" for k in INVARIANTS))
    print(f"   detail: {o.detail or '-'}")


def modal_command(seed: int, fake: str) -> list[str]:
    """redteam_suite invocation that replays the same scenario and the same gold/fake pair as the local demo."""
    cases = ",".join(["gold", fake] if fake else ["gold"])
    return [sys.executable, str(ROOT / "scripts/redteam_suite.py"), "--run", "quickstart", "--seed", str(seed),
            "--cases", cases, "--workers", "2", "--cap-usd", "0.15"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123, help="123 = Task A1, 4 ranks, rank 2 enters a different collective")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fake", default="F11", help="fake repair shown locally and replayed with --modal (F11 = world_size:1; '' = none)")
    ap.add_argument("--modal", action="store_true", help="verified path on the Modal Sandbox backend (costs money)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.out or tempfile.mkdtemp(prefix="traindoctor_qs_"))
    sc = make_scenario(a.seed)
    print(f"== 1. failed job from seed {a.seed}: kind={sc.kind} variant={sc.variant} world_size={sc.world_size}")
    shutil.rmtree(out / f"s{a.seed}", ignore_errors=True)   # a re-run into the same --out starts clean
    snap = execute(sc, out / f"s{a.seed}")
    truth = load_truth(snap)
    print(f"   {len(snap.records)} public records (source=executed). What the agent sees:")
    for r in snap.records:
        if r["kind"] in ("job", "scheduler", "process"):
            print(f"   {r['id']} {r['kind']:9} {r['text']}")
    last = {}
    for r in snap.records:
        if r["kind"] == "trace":
            last[r["rank"]] = r
    for rk in sorted(last):
        r = last[rk]
        print(f"   {r['id']} trace     {r['text']}")
    print(f"   hidden ground truth (never shown to an agent): label={truth.label} {truth.details}")

    print("\n== 2. gold repair (oracle: label, cited evidence ids, source edit)")
    label, evidence, edits = reference_solution(snap, truth)
    print(f"   diagnosis={label} evidence={evidence}")
    show_diff(edits, snap.files)

    print("\n== 3. re-run the repaired job LOCALLY (UNVERIFIED: not sandboxed)")
    o = reexec.verify_fix(snap, edits, allow_unsandboxed=True)
    report("gold, local", o)
    if not o.sandboxed:
        print("   -> the frozen verifier refuses to count this: re-executing agent-edited code without OS/gVisor isolation is not\n"
              "      verification, so ok is forced False (INVALID) although all invariants above hold. Use --modal for the verified path.")
    if a.fake:
        from redteam_suite import build_cases
        cases = {c.id: c for c in build_cases(sc.kind, sc.variant, snap.files, truth.details, edits)}
        if a.fake not in cases:
            sys.exit(f"unknown fake {a.fake!r} for this scenario; available: {', '.join(sorted(cases))}")
        print(f"\n   fake repair {a.fake}: {cases[a.fake].note}")
        report(f"{a.fake}, local", reexec.verify_fix(snap, cases[a.fake].edits, allow_unsandboxed=True))

    if a.modal:
        print("\n== 4. verified path: Modal Sandbox replay (gold must be accepted, the fake rejected)")
        env = dict(os.environ, TD_RESULTS_DIR=str(out / "results"))
        if importlib.util.find_spec("modal") is None and not env.get("TD_MODAL_SITE"):
            sys.exit("the modal client is not importable: pip install modal && modal token new (or set TD_MODAL_SITE to its site-packages)")
        return subprocess.call(modal_command(a.seed, a.fake), env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
