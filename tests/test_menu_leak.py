"""Menu-only leakage (the project rules: leak tests are statistical). Rule: over >= 200 seeds, a lookup learner that sees only
features of the repair menu (no records) must not beat chance at picking the correct repair, nor the majority class at
predicting the label, for the structural features the audit found to identify both on DEV400: which files/keys a
candidate touches, its diff shape and its length. Pure Python (no job is executed)."""
from audit.menu_leak import report, violations

SEEDS = range(10000, 10240)   # 240 seeds; the n-gram predictor is cross-validated over all of them
REP = None


def rep():
    global REP
    REP = REP or report(SEEDS)
    return REP


def test_menu_structure_does_not_identify_the_repair_or_the_label():
    r = rep()
    assert r["n_episodes"] >= 200
    assert not violations(r), violations(r)


def test_diff_text_direction_and_variant_text_are_at_chance():
    """Rule: the correct repair is a property of the incident, not of how the menu is drawn. The display shows both forms
    of a location, in an order fixed by a hash, drawn from the clean job text, so no side or sign marks the repair."""
    v = rep()["repair_leakage"]
    for k in ("content.added_text", "content.removed_text", "content.first_form_text", "content.variant_set_text"):
        assert v[k]["above_chance"] <= v[k]["margin"], k


def test_ngram_logistic_regression_on_menu_text_is_at_chance_across_seeds():
    lr = rep()["ngram_lr"]
    assert lr["n"] >= 200 and lr["above_chance"] <= lr["margin"], lr


def test_boosted_stumps_on_menu_only_features_do_not_beat_majority():
    """Rule: no menu-only predictor beats the majority class by more than its margin on held-out seeds, for the repair
    slot and the label. Regression for 796ba23: the no-change and exclude items drew context from the BUGGY job text
    (`reduce_bcast` in 50/50 A1-per-rank menus), and stumps reached 0.495 vs 0.360. The bundle config alone legitimately
    reveals A1-cfg and A2 (evidence, not a menu leak), so config is not an input."""
    for what, v in rep()["stumps"].items():
        assert v["above_majority"] <= v["margin"], (what, v)


def test_candidate_ids_are_not_a_public_function_of_seed_and_role():
    """Rule: ids are keyed by a secret (TD_KEY), so the old public sha256(seed:menu:role) cannot recover seed or answer;
    a different key gives different ids for the same seed (evidence ids stay key-independent)."""
    import os
    from traindoctor_env.bundle import candidate_id
    assert rep()["id_recovery"]["recovered"] == 0
    a = {candidate_id(10000, r) for r in ("correct", "noop", "exclude_rank")}
    old = os.environ["TD_KEY"]
    try:
        os.environ["TD_KEY"] = old + "-other"
        b = {candidate_id(10000, r) for r in ("correct", "noop", "exclude_rank")}
    finally:
        os.environ["TD_KEY"] = old
    assert a.isdisjoint(b)
    assert candidate_id(10000, "correct") == candidate_id(10000, "correct")


def test_noop_and_exclude_items_carry_no_config_bug_text():
    """Rule (796ba23 / 97a49d8): every item's context is drawn from the CLEAN job, so the no-change relaunch and the
    rank-exclusion items never contain the text of a config bug (a rank's `reduce_bcast` sync mode, a `part-<world>`
    shard dir). Only the repair-location items show the two forms of a location. Direct structural check: the
    statistical tests above did not catch reintroducing the buggy context on 240 seeds (mutation_check_leak.json)."""
    import tempfile
    from dataclasses import asdict
    from pathlib import Path
    from types import SimpleNamespace
    from jobs.build import materialize
    from jobs.config import job_config
    from traindoctor_env.bundle import make_menu
    from traindoctor_env.scenario import Truth, job_spec, make_scenario
    from traindoctor_env.snapshot import job_files
    n = 0
    for seed in SEEDS:
        sc = make_scenario(seed); sp = job_spec(sc)
        tokens = [e.new.strip() for e in sp.bug if e.path == "job/config.yaml"]
        if not tokens:
            continue
        with tempfile.TemporaryDirectory() as t:
            materialize(Path(t) / "jd", sp.config, sp.bug)
            snap = SimpleNamespace(files=job_files(Path(t) / "jd"), config=job_config(Path(t) / "jd"), seed=seed)
        truth = Truth(seed, sc.kind, sc.variant, sc.label, sp.task, [asdict(e) for e in sp.fix], sp.details)
        for c in make_menu(snap, truth):
            if c.role in ("noop", "exclude_rank"):
                n += 1
                for tok in tokens:
                    assert tok not in c.diff(snap.files), (seed, c.role, tok)
    assert n >= 100
