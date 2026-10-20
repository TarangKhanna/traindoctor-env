"""Menu v3 (offline: no Gloo, no Docker, no Modal). Rules: (1) v1 and v2 are unchanged; (2) a v3 menu is a function of the
job's files and the seed only (the Truth is never read), has K_MENU_V3 equally long items, and every item's edits apply to
the job exactly once and leave valid Python; (3) an item yields the same code at its location whatever the job started
with, and with numbers masked every item renders identically in every variant, so the menu alone carries no information
about the label or the repair; (4) the prompt fits the token cap. Which items recover a job is NOT asserted here: that is
decided only by re-execution (the outcome table)."""
import ast
import re
import tempfile
from pathlib import Path

import pytest

from jobs.build import apply_edits
from traindoctor_env.bundle import CAP_TOKENS, make_bundle, make_menu
from traindoctor_env.menu_v3 import ITEM_CHARS_V3, K_MENU_V3
from traindoctor_env.scenario import Scenario
from tests.test_bundle_leak import fake_snapshot

VARIANTS = [("A1", 0), ("A1", 1), ("A2", 0), ("B_SKIP", 0), ("B_DUP", 0), ("B_OK", 0)]
FILES = ("job/train.py", "job/ckpt.py", "job/config.yaml")


def snap_for(kind, variant, seed, world, tmp):
    return fake_snapshot(Scenario(seed, kind, variant, world, 1000 + seed), Path(tmp) / "jd")


def applied(t, c) -> dict:
    d = Path(t) / f"apply_{c.id}"
    (d / "job").mkdir(parents=True)
    for p in FILES:
        (d / p).write_bytes((Path(t) / "jd" / p).read_bytes())
    apply_edits(d, list(c.edits))
    return {p: (d / p).read_text() for p in FILES}


@pytest.mark.parametrize("kind, variant", VARIANTS)
def test_v3_shape_seeded_truth_free_and_edits_apply_once(kind, variant):
    for seed in range(8):
        for world in (2, 3, 4):
            with tempfile.TemporaryDirectory() as t:
                snap, truth = snap_for(kind, variant, seed, world, t)
                menu = make_menu(snap, None, "v3")                      # the Truth is not needed
                assert [c.id for c in menu] == [c.id for c in make_menu(snap, truth, "v3")]
                assert len(menu) == K_MENU_V3 and len({c.id for c in menu}) == K_MENU_V3
                assert {len(c.diff()) for c in menu} == {ITEM_CHARS_V3}
                assert [c.role for c in menu if c.destructive] == ["exclude_rank"]
                for c in menu:
                    out = applied(t, c)
                    for p in ("job/train.py", "job/ckpt.py"):
                        ast.parse(out[p])


def _masked(menu) -> dict:
    """Numbers masked. The shard item marks rank r's line: the stalled rank for A2, a seed-only uniform draw otherwise
    (same distribution); its lines are compared as a multiset so only that position, not the text, may differ."""
    out = {c.role: re.sub(r"\d+", "N", c.diff()) for c in menu}
    out["shard:canonical"] = sorted(re.sub(r"^[> ] ", "", x) for x in out["shard:canonical"].splitlines())
    return out


def test_with_numbers_masked_every_item_renders_identically_in_every_variant():
    """Rule: the menu alone carries no label. Numbers (ranks, config values) are those the shown config already holds."""
    for seed in range(6):
        for world in (3, 4):
            renders = []
            for kind, variant in VARIANTS:
                with tempfile.TemporaryDirectory() as t:
                    renders.append(_masked(make_menu(snap_for(kind, variant, seed, world, t)[0], None, "v3")))
            assert all(r == renders[0] for r in renders), (seed, world)


def test_each_code_item_yields_the_same_code_whatever_the_job_started_with():
    def finals(kind, variant):
        with tempfile.TemporaryDirectory() as t:
            snap, _ = snap_for(kind, variant, 5, 3, t)
            return {c.role: applied(t, c) for c in make_menu(snap, None, "v3")}
    base = finals("B_OK", 0)
    for kind, variant in VARIANTS:
        other = finals(kind, variant)
        for role in base:
            if role.startswith("sync:"):
                assert other[role]["job/train.py"] == base[role]["job/train.py"], (kind, role)
            if role.startswith("ckpt:"):
                fn = lambda s: re.sub(r"def _progress.*?\n\n\n", "", s, flags=re.S)   # noqa: E731  (B_SKIP's unused helper)
                if role == "ckpt:from_step":    # edits restore() only; the save line stays as the job has it, by design
                    fn = lambda s: s.split("def restore(")[1]   # noqa: E731
                assert fn(other[role]["job/ckpt.py"]) == fn(base[role]["job/ckpt.py"]), (kind, role)


def test_v3_bundles_fit_the_token_cap_for_every_variant_at_3_and_4_ranks():
    from tests.synth_snapshot import synth_snapshot
    from traindoctor_env.scenario import make_scenario
    seen = set()
    for s in range(10000, 10120):
        sc = make_scenario(s)
        with tempfile.TemporaryDirectory() as t:
            snap, truth = synth_snapshot(sc, Path(t) / "jd")
            b = make_bundle(snap, make_menu(snap, truth, "v3"), truth)
            assert b.tokens <= CAP_TOKENS and "marked '>'" in b.text
            seen.add((sc.kind, sc.world_size))
    assert {k for k, _ in seen} >= {"A1", "A2", "B_SKIP", "B_DUP", "B_OK"} and {w for _, w in seen} == {3, 4}


@pytest.mark.parametrize("kind, variant", VARIANTS)
def test_every_displayed_line_is_true_of_the_edited_job(kind, variant):
    """Rule: the display describes the job AFTER the edit. Every '>' and unmarked line it shows is a line of the edited
    file, and every '-' line is gone from it, in every variant (a display that shows a fixed value the edit leaves faulty
    would misdescribe the repair)."""
    for seed in range(4):
        for world in (2, 3, 4):
            with tempfile.TemporaryDirectory() as t:
                snap, _ = snap_for(kind, variant, seed, world, t)
                for c in make_menu(snap, None, "v3"):
                    if c.role == "relaunch":
                        continue
                    out, path = applied(t, c), None
                    for line in c.diff().splitlines():
                        if line.startswith("--- "):
                            path = line[4:]
                        elif line[:2] in ("> ", "  "):
                            assert line[2:] in out[path].splitlines(), (kind, seed, world, c.role, line)
                        elif line[:2] == "- ":
                            assert line[2:] not in out[path].splitlines(), (kind, seed, world, c.role, line)
