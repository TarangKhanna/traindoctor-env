"""Local spend guard. Ledger (modal_train/ledger.jsonl, gitignored) rows: {ts, fn, gpu, seconds, rate, usd, kind}.
`kind` is "projected" (written at launch, worst case = function timeout) or "actual" (written from the returned result,
which supersedes the projection with the same run_id). Pure Python; no Modal import."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
RATES = HERE / "rates.json"
CAP_USD = float(os.environ.get("TD_TASK_CAP", "30.0"))  # task-local cap override, e.g. TD_TASK_CAP=1.5


class BudgetExceeded(RuntimeError):
    pass


def rate_per_s(gpu: str, rates_path: Path = RATES) -> float:
    r = json.loads(rates_path.read_text())
    if gpu not in r:
        raise KeyError(f"no rate configured for {gpu!r} in {rates_path}")
    return float(r[gpu]) / 3600.0


def _rows(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]


def spent_usd(ledger: Path = LEDGER) -> float:
    """Total: actual rows count; a projected row counts only if its run_id has no actual row."""
    rows = _rows(ledger)
    actual = {r["run_id"] for r in rows if r["kind"] == "actual"}
    return sum(r["usd"] for r in rows if r["kind"] == "actual" or r["run_id"] not in actual)


def project(gpu: str, seconds: float, ledger: Path = LEDGER, rates_path: Path = RATES, cap: float = CAP_USD) -> float:
    """Return projected total after this run; raise BudgetExceeded if it is above cap."""
    add = rate_per_s(gpu, rates_path) * seconds
    total = spent_usd(ledger) + add
    if total > cap:
        raise BudgetExceeded(f"projected ${total:.2f} > cap ${cap:.2f} (ledger ${spent_usd(ledger):.2f} + run ${add:.2f})")
    return total


def record(run_id: str, fn: str, gpu: str, seconds: float, kind: str, ledger: Path = LEDGER, rates_path: Path = RATES) -> dict:
    assert kind in ("projected", "actual")
    rate = rate_per_s(gpu, rates_path) * 3600.0
    row = dict(ts=time.time(), run_id=run_id, fn=fn, gpu=gpu, seconds=seconds, rate=rate, usd=rate / 3600.0 * seconds, kind=kind)
    with open(ledger, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row
