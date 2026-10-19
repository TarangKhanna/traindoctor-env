"""Case-list sanity for scripts/modal_regression.py (no Modal). Rule: the regression suite must cover every task kind with
a gold (accepted), a null (rejected) and >= 2 incorrect repairs (rejected), on seeds no earlier experiment used."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import modal_regression as mr  # noqa: E402

from traindoctor_env.scenario import make_scenario  # noqa: E402

EARLIER = [range(10000, 10400), range(20000, 20400), range(7001, 7009), range(5000, 5800), range(7000, 7200),
           range(40000, 40100), range(60000, 70000), range(70000, 71010), range(80000, 81000), (101, 201, 202)]


def test_five_kinds_covered_with_expected_roles():
    assert set(mr.CASES) == {"A1_src", "A1_cfg", "A2", "B_SKIP", "B_DUP"}
    for kind, cs in mr.CASES.items():
        assert [c for c, ok in cs.items() if ok] == ["gold"], kind
        assert cs.get("null") is False
        assert sum(1 for c, ok in cs.items() if not ok and c != "null") >= 2, kind


def test_a_kinds_include_world_size_shortcut():
    for k in ("A1_src", "A1_cfg", "A2"):
        assert "F11" in mr.CASES[k]


def test_seeds_build_the_named_kind_and_are_disjoint_from_earlier_lists():
    for name, seed in mr.SEEDS.items():
        assert 90000 <= seed <= 90999
        assert not any(seed in r for r in EARLIER), seed
        sc = make_scenario(seed)
        assert (sc.kind, sc.variant) == mr.KIND_OF[name]
    assert len(set(mr.SEEDS.values())) == 5
