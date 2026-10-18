"""Per-task validation under trusted sandbox replay (backend='modal'). Writes receipts/<tid>.json.

  python -m pilot.validate run [--tasks p01,p02] [--arms noop,gold,decoy,alt] [--workers 8]
  python -m pilot.validate summary            # table + dedup test from the receipts
Checks (each must be able to fail):
  broken_fails   no-op replay of the broken job is NOT a verified recovery
  gold_pass      reverse patch IS a verified recovery
  decoy          decoy COMPLETES (final checkpoint committed) and FAILS the recovery invariants
  alt_pass       alternative valid fix is a verified recovery (null where no alternative exists)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pilot import core  # noqa: E402
from generator.mutate import arm_edits  # noqa: E402

RECEIPTS = core.HERE / "receipts"


def run_arm(tid: str, arm: str) -> tuple:
    snap = core.build_snapshot(tid)
    m = core.mutant(tid)
    edits = arm_edits(m, arm)
    if arm == "alt" and not edits:
        return tid, arm, None
    return tid, arm, core.live_outcome(tid, snap, edits, tag=arm)


def ledger_cost(tid: str) -> float:
    from modal_train import budget
    rows = [json.loads(x) for x in budget.LEDGER.read_text().splitlines() if x.strip()]
    return sum(r["usd"] for r in rows if str(r.get("run_id", "")).startswith(f"{core.LEDGER_PREFIX}{tid}-") and r["kind"] == "actual")


def receipt(tid: str, res: dict) -> dict:
    snap = core.build_snapshot(tid)
    m = core.mutant(tid)
    ok = lambda a: res.get(a) is not None and core.verified(res[a])  # noqa: E731
    out = lambda a: None if res.get(a) is None else res[a].get("outcome")  # noqa: E731
    dec = out("decoy")
    checks = dict(
        broken_fails=res["noop"]["status"] == "ok" and not ok("noop"),
        noop_fails=res["noop"]["status"] == "ok" and not ok("noop"),
        gold_pass=ok("gold"),
        decoy_completes_and_fails=bool(dec) and bool(dec["ckpt_valid"]) and not dec["ok"] and res["decoy"]["status"] == "ok",
        alt_pass=(ok("alt") if m.alt else None),
    )
    all_ok = all(v for v in checks.values() if v is not None)
    fp = core.fingerprint(snap, out("noop"))
    slim = lambda a: None if res.get(a) is None else dict(  # noqa: E731
        status=res[a]["status"], verified=ok(a), wall_s=round(res[a].get("wall_s", 0), 1),
        failing_invariants=core.failing_invariants(out(a)) if out(a) else None,
        ok=out(a)["ok"] if out(a) else None, path=out(a)["path"] if out(a) else None, lost_steps=out(a)["lost_steps"] if out(a) else None,
        detail=out(a)["detail"] if out(a) else res[a].get("error"), exec_mode=out(a)["exec_mode"] if out(a) else None)
    return dict(task=tid, seed=m.seed, operator=m.op, variant=m.variant, kind=m.kind, mutant_sha256=m.digest(),
                edits_key={a: core.edits_key(arm_edits(m, a)) for a in ("noop", "gold", "decoy", "alt")},
                decoy_note=m.decoy_note, alt_note=m.alt_note, incident=dict(meta=snap.meta, n_records=len(snap.records)),
                arms={a: slim(a) for a in ("noop", "gold", "decoy", "alt")}, checks=checks, all_checks_pass=all_ok,
                fingerprint=fp, modal_cost_usd=round(ledger_cost(tid), 4))


def cmd_run(tasks, arms, workers):
    RECEIPTS.mkdir(exist_ok=True)
    jobs = [(t, a) for t in tasks for a in arms]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda ta: run_arm(*ta), jobs))
    res: dict = {}
    for tid, arm, r in results:
        res.setdefault(tid, {})[arm] = r
        print(tid, arm, None if r is None else (r["status"], r.get("outcome", {}).get("ok"), round(r["wall_s"]), "cached" if r.get("cached") else ""), flush=True)
    print(f"{len(jobs)} replays in {time.time() - t0:.0f}s; mutpilot spend so far ${core.spent_usd():.4f}", flush=True)
    for tid in tasks:
        if all(a in res[tid] for a in ("noop", "gold", "decoy", "alt")):
            (RECEIPTS / f"{tid}.json").write_text(json.dumps(receipt(tid, res[tid]), indent=1, sort_keys=True, default=str))


def load_receipts() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RECEIPTS.glob("p*.json"))}


def summary() -> str:
    rs = load_receipts()
    L = ["| task | op | kind | broken fails | gold PASS | decoy completes+fails | alt PASS | all checks | mechanism fp | full fp | $ |", "|" + "---|" * 11]
    for tid, r in rs.items():
        c = r["checks"]
        f = lambda v: "n/a" if v is None else ("yes" if v else "NO")  # noqa: E731
        L.append(f"| {tid} | {r['operator']}{r['variant']} | {r['kind']} | {f(c['broken_fails'])} | {f(c['gold_pass'])} | {f(c['decoy_completes_and_fails'])} | "
                 f"{f(c['alt_pass'])} | {f(r['all_checks_pass'])} | {r['fingerprint']['mechanism_hash']} | {r['fingerprint']['full_hash']} | {r['modal_cost_usd']} |")
    d = core.dedup({t: r["fingerprint"] for t, r in rs.items()})
    L += ["", f"tasks passing every applicable check: {sum(r['all_checks_pass'] for r in rs.values())}/{len(rs)}",
          f"unique behavioural fingerprints: mechanism-level {d['n_unique_mechanism']}/{d['n_tasks']}, full {d['n_unique_full']}/{d['n_tasks']}",
          f"mechanism-level groups (equal fingerprint = one task): {d['mechanism_groups']}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--arms", default="noop,gold,decoy,alt")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = sorted(core.TASKS) if a.tasks == "all" else a.tasks.split(",")
    if a.cmd == "run":
        cmd_run(tasks, a.arms.split(","), min(a.workers, 8))
    else:
        s = summary()
        (core.OUT / "validation_summary.md").write_text(s)
        print(s)


if __name__ == "__main__":
    main()
