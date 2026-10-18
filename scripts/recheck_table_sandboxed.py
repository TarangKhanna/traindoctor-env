"""Re-execute cached fix-outcome table cells UNDER THE OS SANDBOX and compare with the cached dev_container outcomes.
Uses the real verification path (traindoctor_env.reexec.verify_fix with allow_unsandboxed=False); never the
dev_container mode. Fails closed if sandbox-exec cannot be applied. Writes a results JSON; prints agree/disagree per
kind x repair role. The pickles are this project's own (sha256 checked against modal_train/cache/PROVENANCE.md).
  recheck_table_sandboxed.py --episodes dev|train [--seed 0] [--out PATH] [--dry-select]"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "modal_train" / "cache"
MIN_TOTAL, ALL_BELOW, MIN_NONREC = 60, 300, 40


def select_cells(cells: list[dict], seed: int = 0, min_total: int = MIN_TOTAL, all_below: int = ALL_BELOW,
                 min_nonrec: int = MIN_NONREC) -> list[dict]:
    """cells: dicts with keys ep, kind, role, rid, cached_ok. All if fewer than `all_below`. Otherwise every recovering
    cell plus a seeded sample of non-recovering ones, stratified by (kind, role) round-robin so every stratum is
    represented; the total is at least `min_total`."""
    if len(cells) < all_below:
        return list(cells)
    rec = [c for c in cells if c["cached_ok"]]
    non = [c for c in cells if not c["cached_ok"]]
    n_non = min(len(non), max(min_total - len(rec), min_nonrec))
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for c in non:
        strata[(c["kind"], c["role"])].append(c)
    queues = []
    for key in sorted(strata):
        q = sorted(strata[key], key=lambda c: (c["ep"], c["rid"]))
        random.Random(f"{seed}:{key}").shuffle(q)
        queues.append(q)
    picked: list[dict] = []
    while len(picked) < n_non:
        for q in queues:
            if q and len(picked) < n_non:
                picked.append(q.pop())
    return rec + picked


def cells_of(eps: list) -> list[dict]:
    return [dict(ep=i, kind=truth.kind, role=next(c.role for c in menu if c.id == rid), rid=rid,
                 cached_ok=bool(o.ok), cached_fault_not_injected=bool(o.fault_not_injected))
            for i, (truth, _b, menu, table) in enumerate(eps) for rid, o in table.items()]


def _check_provenance(name: str) -> None:
    pins = os.environ.get("TD_PINS")
    if pins:   # installed tdrun run: only the owner manifest decides; PROVENANCE.md (agent-writable worktree) is not trusted
        want = json.loads(Path(pins).read_text())["files"].get(f"modal_train/cache/{name}")
        got = hashlib.sha256((CACHE / name).read_bytes()).hexdigest()
        if want is None or want != got:
            sys.exit(f"{name}: sha256 {got} not in the owner manifest ({want}); refusing to unpickle")
        return
    prov = (CACHE / "PROVENANCE.md").read_text()
    want = next((ln.split()[0] for ln in prov.splitlines() if ln.strip().endswith(name)), None)
    got = hashlib.sha256((CACHE / name).read_bytes()).hexdigest()
    if want != got:
        sys.exit(f"{name}: sha256 {got} != PROVENANCE {want}; refusing to unpickle")


def summarize(rows: list[dict]) -> str:
    agg: dict[tuple, Counter] = defaultdict(Counter)
    for r in rows:
        k = (r["kind"], r["role"])
        agg[k]["n"] += 1
        agg[k]["agree" if r["agree"] else "disagree"] += 1
        agg[k]["invalid"] += r.get("live_sandboxed") is False
        agg[k]["error"] += bool(r.get("error"))
    lines = [f"{'kind':6} {'repair':16} {'n':>4} {'agree':>6} {'disagree':>8} {'invalid':>7} {'error':>5}"]
    for (kind, role), c in sorted(agg.items()):
        lines.append(f"{kind:6} {role:16} {c['n']:>4} {c['agree']:>6} {c['disagree']:>8} {c['invalid']:>7} {c['error']:>5}")
    t = Counter()
    for c in agg.values():
        t.update(c)
    lines.append(f"{'TOTAL':23} {t['n']:>4} {t['agree']:>6} {t['disagree']:>8} {t['invalid']:>7} {t['error']:>5}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", choices=["dev", "train"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-select", action="store_true", help="print the selection and stop (no execution)")
    a = p.parse_args(argv)
    _check_provenance(f"{a.episodes}_eps.pkl")
    eps = pickle.loads((CACHE / f"{a.episodes}_eps.pkl").read_bytes())
    sel = select_cells(cells_of(eps), a.seed)
    print(f"{a.episodes}: {len(eps)} episodes, {sum(len(e[3]) for e in eps)} cells; selected {len(sel)} "
          f"({sum(c['cached_ok'] for c in sel)} recovering)", flush=True)
    if a.dry_select:
        return 0
    from jobs import sandbox
    from traindoctor_env import reexec
    from traindoctor_env.scenario import make_scenario
    from traindoctor_env.snapshot import execute
    if not sandbox.available():
        sys.exit("sandbox-exec cannot be applied here; refusing to re-execute (fail closed)")
    assert not reexec.DEV_CONTAINER_MODE and not reexec.ALLOW_UNSANDBOXED, "unsandboxed mode must be off"
    out_path = a.out or Path(os.environ.get("TD_RESULTS_DIR") or ROOT / "modal_train" / "results") / f"recheck_sandboxed_{a.episodes}.json"
    by_ep: dict[int, list[dict]] = defaultdict(list)
    for c in sel:
        by_ep[c["ep"]].append(c)
    rows: list[dict] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as wd:
        for ep in sorted(by_ep):
            truth, _b, menu, table = eps[ep]
            try:
                snap = execute(make_scenario(truth.seed), Path(wd) / str(truth.seed))
                snap_err = None
            except Exception as e:  # noqa: BLE001 -- recorded, not hidden
                snap, snap_err = None, f"{type(e).__name__}: {e}"
            for c in by_ep[ep]:
                row = dict(c, seed=truth.seed)
                cached = table[c["rid"]]
                row["cached_exec_mode"] = cached.exec_mode
                if snap is None:
                    row.update(error="snapshot: " + snap_err, agree=False)
                else:
                    try:
                        cand = next(m for m in menu if m.id == c["rid"])
                        live = reexec.verify_fix(snap, list(cand.edits), allow_unsandboxed=False)
                        row.update(live_ok=bool(live.ok), live_sandboxed=bool(live.sandboxed), live_exec_mode=live.exec_mode,
                                   live_path=live.path, live_lost_steps=live.lost_steps, live_detail=live.detail,
                                   live_fault_not_injected=bool(live.fault_not_injected),
                                   agree=bool(live.sandboxed and live.ok == cached.ok))
                    except Exception as e:  # noqa: BLE001
                        row.update(error=f"{type(e).__name__}: {e}", agree=False)
                rows.append(row)
                print(f"[{len(rows)}/{len(sel)} {time.time() - t0:.0f}s] seed={truth.seed} {c['kind']} {c['role']} "
                      f"cached_ok={c['cached_ok']} live_ok={row.get('live_ok')} agree={row['agree']}", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(dict(episodes=a.episodes, seed=a.seed, selected=len(sel), done=len(rows),
                                                complete=False, rows=rows), indent=1, sort_keys=True))
    out_path.write_text(json.dumps(dict(episodes=a.episodes, seed=a.seed, selected=len(sel), done=len(rows),
                                        complete=True, rows=rows), indent=1, sort_keys=True))
    print(summarize(rows))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
