"""Rules for the sandboxed-recheck cell selection (execution is not tested here): every recovering cell is re-checked;
small tables are checked whole; at least 60 cells otherwise; every (kind, role) stratum with non-recovering cells is
represented; selection is a seeded, duplicate-free subset."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("recheck", Path(__file__).resolve().parent.parent / "scripts" / "recheck_table_sandboxed.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)


def mk(n_ep=64, k=6):
    roles = ["correct", "noop", "exclude_rank", "decoy_a", "decoy_b", "decoy_c"]
    return [dict(ep=e, kind=("A1", "A2", "B")[e % 3], role=roles[i], rid=f"{e}-{i}", cached_ok=(i == 0))
            for e in range(n_ep) for i in range(k)]


def key(c): return (c["ep"], c["rid"])


def test_small_table_checked_whole():
    cells = mk(20, 6)  # 120 < 300
    assert rc.select_cells(cells) == cells


def test_all_recovering_and_min_total():
    cells = mk()
    sel = rc.select_cells(cells, seed=3)
    assert {key(c) for c in cells if c["cached_ok"]} <= {key(c) for c in sel}
    assert len(sel) >= 60 and len({key(c) for c in sel}) == len(sel) and {key(c) for c in sel} <= {key(c) for c in cells}


def test_min_total_with_few_recovering():
    cells = [dict(c, cached_ok=(c["ep"] == 0 and c["role"] == "correct")) for c in mk()]
    assert len(rc.select_cells(cells)) >= 60


def test_every_nonrecovering_stratum_represented():
    cells = mk()
    sel = rc.select_cells(cells)
    want = {(c["kind"], c["role"]) for c in cells if not c["cached_ok"]}
    assert want <= {(c["kind"], c["role"]) for c in sel}


def test_seeded_deterministic_and_seed_sensitive():
    cells = mk()
    assert rc.select_cells(cells, 1) == rc.select_cells(cells, 1)
    assert {key(c) for c in rc.select_cells(cells, 1)} != {key(c) for c in rc.select_cells(cells, 2)}


def test_summarize_counts():
    rows = [dict(kind="A1", role="correct", agree=True, live_sandboxed=True), dict(kind="A1", role="correct", agree=False, live_sandboxed=False)]
    assert "TOTAL" in rc.summarize(rows) and "   2" in rc.summarize(rows)
