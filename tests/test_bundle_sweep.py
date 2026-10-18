"""Bundle validity over every task variant x 5 seeds: the bundle fits CAP_TOKENS, keeps the whole evidence
certificate, and the oracle can answer from it alone. Slow (30 executions): run with `pytest -m sweep -s`."""
import statistics
from concurrent.futures import ThreadPoolExecutor

import pytest

from traindoctor_env.bundle import CAP_TOKENS, BundleTooLarge, make_bundle, make_menu
from traindoctor_env.oracle import reference_solution
from traindoctor_env.scenario import Scenario
from traindoctor_env.snapshot import execute, load_truth
from traindoctor_env.verifiers import certificate_ids, evidence_ok

pytestmark = [pytest.mark.sweep, pytest.mark.executed]
VARIANTS = [("A1", 0), ("A1", 1), ("A2", 0), ("B_SKIP", 0), ("B_DUP", 0), ("B_OK", 0)]
SEEDS = [1000, 1001, 1002, 1003, 1004]


def test_every_variant_and_seed_fits_and_is_solvable_from_the_bundle(tmp_path):
    jobs = [(k, v, s) for k, v in VARIANTS for s in SEEDS]
    with ThreadPoolExecutor(2) as ex:
        snaps = list(ex.map(lambda j: execute(Scenario(j[2], j[0], j[1], 2, 5000 + j[2]), tmp_path / f"{j[0]}{j[1]}_{j[2]}"), jobs))
    sizes: dict[str, list[int]] = {}
    too_large = []
    for (kind, var, seed), snap in zip(jobs, snaps):
        truth = load_truth(snap)
        menu = make_menu(snap, truth)
        try:
            bundle = make_bundle(snap, menu, truth)
        except BundleTooLarge as e:
            too_large.append(str(e))
            continue
        shown = {r["id"] for r in bundle.records}
        _, evidence, edits = reference_solution(snap, truth)
        assert certificate_ids(list(snap.records), truth) <= shown
        assert evidence_ok([list(bundle.records)], truth, evidence, shown)
        assert any(c.id in bundle.text for c in menu if c.role == "correct")
        assert bundle.tokens <= CAP_TOKENS
        sizes.setdefault(f"{kind}{var}", []).append(bundle.tokens)
    allv = [t for v in sizes.values() for t in v]
    for k, v in sizes.items():
        print(f"bundle tokens {k}: min {min(v)} median {statistics.median(v)} max {max(v)}")
    print(f"bundle tokens ALL: min {min(allv)} median {statistics.median(allv)} max {max(allv)} (n={len(allv)}, cap {CAP_TOKENS})")
    assert not too_large, too_large
