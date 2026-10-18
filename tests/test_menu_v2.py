"""Menu v2 (offline: no Gloo, no Docker, no Modal). Rules: (1) v1 stays the default and byte-for-byte what it was; (2) a v2 menu
has K_MENU_V2 equally long items, exactly one correct repair, and every candidate's edits apply to the job exactly once and
leave valid Python; (3) the three extra items are the same repair text in every variant (their presence is not evidence), and
menu structure alone does not identify the repair or the label (statistical audit, as for v1); (4) the prompt still fits the
3,072-token cap for every variant. Which extras complete the job and which invariant they break is NOT asserted here: that is
decided by real re-execution (the outcome table)."""
import ast
import tempfile
from pathlib import Path

import pytest

from audit.menu_leak import report, violations
from jobs.build import apply_edits, materialize
from traindoctor_env.bundle import CAP_TOKENS, ITEM_CHARS, K_MENU, K_MENU_V2, make_bundle, make_menu
from traindoctor_env.scenario import Scenario
from tests.test_bundle_leak import fake_snapshot

VARIANTS = [("A1", 0), ("A1", 1), ("A2", 0), ("B_SKIP", 0), ("B_DUP", 0), ("B_OK", 0)]
EXTRA = {"distractor_timeout", "distractor_scratch", "distractor_align"}


def snap_for(kind, variant, seed, world, tmp):
    return fake_snapshot(Scenario(seed, kind, variant, world, 1000 + seed), Path(tmp) / "jd")


@pytest.mark.parametrize("kind, variant", VARIANTS)
def test_v1_is_the_default_and_unchanged(kind, variant):
    with tempfile.TemporaryDirectory() as t:
        snap, truth = snap_for(kind, variant, 7, 3, t)
        a, b = make_menu(snap, truth), make_menu(snap, truth, "v1")
        assert a == b and len(a) == K_MENU and not EXTRA & {c.role for c in a}


@pytest.mark.parametrize("kind, variant", VARIANTS)
def test_v2_menu_shape_one_correct_equal_items_edits_apply_once(kind, variant):
    for seed in range(12):
        for world in (3, 4):
            with tempfile.TemporaryDirectory() as t:
                snap, truth = snap_for(kind, variant, seed, world, t)
                menu = make_menu(snap, truth, "v2")
                assert len(menu) == K_MENU_V2 and len({c.id for c in menu}) == K_MENU_V2
                assert [c.id for c in make_menu(snap, truth, "v2")] == [c.id for c in menu]          # seeded
                assert {c.role for c in menu} >= EXTRA and sum(c.role == "correct" for c in menu) == 1
                assert {len(c.diff(snap.files)) for c in menu} == {ITEM_CHARS}
                assert not any(c.destructive for c in menu if c.role in EXTRA)
                for c in menu:                  # each edit set applies to the incident's job exactly once and stays valid Python
                    d = Path(t) / f"apply_{c.id}"
                    spec_files = {p: (Path(t) / "jd" / p).read_bytes() for p in ("job/train.py", "job/ckpt.py", "job/config.yaml")}
                    (d / "job").mkdir(parents=True)
                    for p, b in spec_files.items():
                        (d / p).write_bytes(b)
                    apply_edits(d, list(c.edits))
                    for p in ("job/train.py", "job/ckpt.py"):
                        ast.parse((d / p).read_text())


@pytest.mark.parametrize("kind, variant", VARIANTS)
def test_extras_are_the_same_repair_text_in_every_variant(kind, variant):
    """Rule: an item present in all variants carries no label. Compare the post-edit text of each extra across variants that
    share the clean files: timeout and scratch edit identical lines everywhere; align yields identical sync code everywhere."""
    def finals(k, v):
        with tempfile.TemporaryDirectory() as t:
            snap, truth = snap_for(k, v, 5, 3, t)
            out = {}
            for c in make_menu(snap, truth, "v2"):
                if c.role in EXTRA:
                    d = Path(t) / f"x_{c.role}"
                    (d / "job").mkdir(parents=True)
                    for p in ("job/train.py", "job/ckpt.py", "job/config.yaml"):
                        (d / p).write_bytes((Path(t) / "jd" / p).read_bytes())
                    apply_edits(d, list(c.edits))
                    out[c.role] = {p: (d / p).read_text() for p in ("job/train.py", "job/ckpt.py")}
            return out
    base, other = finals("B_OK", 0), finals(kind, variant)
    # align: the final sync loop is the same code whether the job started clean or held the rank-keyed branch (A1 src, which
    # is the only variant whose train.py differs from the pristine one)
    assert other["distractor_align"]["job/train.py"] == base["distractor_align"]["job/train.py"]
    def edits(k, v):                               # timeout and scratch: the very same edit in every variant
        with tempfile.TemporaryDirectory() as t:
            return {c.role: c.edits for c in make_menu(*snap_for(k, v, 5, 3, t), "v2") if c.role in EXTRA}
    a, b = edits("B_OK", 0), edits(kind, variant)
    for role in ("distractor_timeout", "distractor_scratch"):
        assert a[role] == b[role], role


def test_v2_menu_structure_does_not_identify_the_repair_or_the_label():
    """Rule (the project rules: leak tests are statistical): over 240 seeds a menu-only learner is at chance for repair and label."""
    rep = report(range(10000, 10240), make_menu_fn=lambda s, t: make_menu(s, t, "v2"))
    assert rep["n_episodes"] >= 200
    assert not violations(rep), violations(rep)


def test_v2_bundles_fit_the_token_cap_for_every_variant_at_3_and_4_ranks():
    from tests.synth_snapshot import synth_snapshot
    from traindoctor_env.scenario import make_scenario
    seen = set()
    for s in range(10000, 10120):
        sc = make_scenario(s)
        with tempfile.TemporaryDirectory() as t:
            snap, truth = synth_snapshot(sc, Path(t) / "jd")
            b = make_bundle(snap, make_menu(snap, truth, "v2"), truth)       # raises BundleTooLarge if the certificate cannot fit
            assert b.tokens <= CAP_TOKENS
            seen.add((sc.kind, sc.world_size))
    assert {k for k, _ in seen} >= {"A1", "A2", "B_SKIP", "B_DUP", "B_OK"} and {w for _, w in seen} == {3, 4}
