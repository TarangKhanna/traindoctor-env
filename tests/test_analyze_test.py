"""scripts/analyze_test.py on synthetic data. Rules asserted (docs/prereg/test_protocol.md): the outcome class follows the CI vs
MMD rule; the primary contrast is V2-arm minus V0-arm on the 4 faulted kinds only (B_OK excluded); invalid or errored replays
are never successes; claimed-but-unverified counts valid claimed rollouts that are not V2; t critical value matches the table."""
import math

import numpy as np

from scripts.analyze_test import KINDS, SEEDS, analyse, classify, markdown, paired_t, t_cdf, t_crit

G = 4


def synth(p_v2arm, p_v0arm, n_per_kind=6, seed=0, base=0.0):
    """Policies: base/sft/csft constant; v0_s*/v2_s* succeed (V2) on a fixed fraction of rollouts per episode."""
    rng = np.random.default_rng(seed)
    rolls = []
    kinds = [KINDS[i % 5] for i in range(5 * n_per_kind)]
    for pol in ["base", "sft", "csft"] + [f"v0_s{s}" for s in SEEDS] + [f"v2_s{s}" for s in SEEDS]:
        p = p_v2arm if pol.startswith("v2_") else p_v0arm if pol.startswith("v0_") else base
        for e, k in enumerate(kinds):
            for g in range(G):
                ok = rng.random() < p
                rolls.append(dict(policy=pol, ep=e, kind=k, valid=True, v0=True, v1=ok, v2=ok, claimed=True, replay="ok", unsafe=False))
    return dict(rollouts=rolls)


def test_t_distribution_matches_table_values():
    assert abs(t_crit(5) - 2.5706) < 1e-3          # standard two-sided 95% critical value, df 5
    assert abs(t_cdf(0.0, 5) - 0.5) < 1e-9
    assert abs(2 * (1 - t_cdf(2.2381, 5)) - 0.0754) < 2e-3


def test_paired_t_known_sample():
    r = paired_t([1.8, 5.2, 6.0, 10.2, 8.6, -3.9])   # the 6-seed DEV differences quoted in docs/RELEASE_MANIFEST.md (row 8)
    assert abs(r["mean"] - 4.6) < 0.05 and abs(r["t"] - 2.24) < 0.05 and abs(r["p"] - 0.075) < 0.01
    assert r["ci95"][0] < 0 < r["ci95"][1]


def test_outcome_class_rule_is_ci_vs_mmd():
    assert classify(8.0, 15.0).startswith("effect >= minimum")
    assert classify(1.0, 5.0).startswith("practically insignificant")
    assert classify(-3.0, 4.0).startswith("falsified")
    assert classify(-9.0, -1.0).startswith("falsified") and "worse" in classify(-9.0, -1.0)
    assert classify(-2.0, 12.0).startswith("inconclusive")
    assert classify(2.0, 12.0).startswith("positive effect, size unresolved")


def test_identical_arms_give_zero_difference_and_not_a_win():
    res = analyse(synth(0.4, 0.4, seed=1))
    lo, hi = res["primary"]["boot_ci95"]
    assert lo < 0 < hi
    assert "inconclusive" in res["primary"]["outcome_class"] or "falsified" in res["primary"]["outcome_class"]


def test_large_true_gap_is_detected():
    res = analyse(synth(0.9, 0.1, n_per_kind=10, seed=2))
    assert res["primary"]["diff"] > 60 and res["primary"]["boot_ci95"][0] > 50
    assert res["primary"]["outcome_class"].startswith("effect >= minimum")


def test_primary_excludes_b_ok_but_pooled_sensitivity_includes_it():
    d = synth(0.5, 0.5, seed=3)
    for r in d["rollouts"]:                         # make V2-arm succeed ONLY on B_OK
        if r["policy"].startswith("v2_"):
            r["v2"] = r["kind"] == "B_OK"
        if r["policy"].startswith("v0_"):
            r["v2"] = False
    res = analyse(d)
    assert res["primary"]["diff"] == 0.0            # faulted kinds only: arms identical (both 0)
    assert res["primary_all5_pooled"]["diff"] > 0   # B_OK difference appears only in the pooled sensitivity view


def test_claimed_not_verified_and_replay_errors_are_counted_not_hidden():
    d = synth(0.5, 0.5, seed=4)
    d["rollouts"][0].update(replay="error", v2=False, claimed=True)
    d["rollouts"][1].update(valid=False, v0=False, v1=False, v2=False, claimed=False, replay="none")
    res = analyse(d)
    assert res["replay_errors_total"] == 1
    pol = res["policy_table"][d["rollouts"][0]["policy"]]
    assert pol["claimed_not_verified"] == pol["claimed"] - pol["claimed_and_verified"]
    assert pol["valid"] < 1.0 and pol["replay_errors"] == 1


def test_missing_episode_for_a_policy_fails_loudly():
    d = synth(0.5, 0.5)
    d["rollouts"] = [r for r in d["rollouts"] if not (r["policy"] == "v0_s3" and r["ep"] == 2)]
    try:
        analyse(d)
    except AssertionError as e:
        assert "v0_s3" in str(e)
    else:
        raise AssertionError("expected AssertionError")


def test_markdown_renders():
    assert "outcome class" in markdown(analyse(synth(0.5, 0.4, seed=5)))
