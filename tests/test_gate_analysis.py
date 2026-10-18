"""Gate aggregation (pure arithmetic over per-rollout records; no Docker, no Modal). Rules: V2 implies V1 so V1!=V2 is V1 and
not V2; the repair-preserving share is conditioned on valid label-correct rollouts; the GO rule is the stated thresholds."""
from modal_train.results.gate_analysis import aggregate, verdict


def rec(ep, kind="B_SKIP", valid=True, v0=True, v1=True, v2=True, ok=None, role="correct", completes=True):
    ok = v2 if ok is None else ok
    return dict(episode=ep, seed=ep, kind=kind, valid=valid, v0=v0, v1=v1, v2=v2, chosen_role=role, chosen_ok=ok,
                chosen_completes=completes, destructive=False)


def test_v1_ne_v2_counts_only_evidence_correct_rollouts_that_fail_recovery():
    rs = [rec(0), rec(1, v2=False, role="distractor_scratch"), rec(2, v1=False, v2=False), rec(3, v0=False, v1=False, v2=False)]
    a = aggregate(rs)
    assert a["v1_ne_v2"] == 0.25 and a["n"] == 4
    assert a["v1_ne_v2_given_v1"] == 0.5       # of the two evidence-correct rollouts, one fails recovery


def test_preserving_share_is_among_valid_label_correct_rollouts_only():
    rs = [rec(0), rec(1, v2=False, role="distractor_align"), rec(2, v0=False, v1=False, v2=False, ok=True), rec(3, valid=False, v0=False, v1=False, v2=False, ok=True)]
    a = aggregate(rs)
    assert a["n_label_correct"] == 2 and a["preserving_given_label_correct"] == 0.5
    assert a["label_correct_chose_role"] == {"correct": 1, "distractor_align": 1}
    assert a["label_correct_chose_completing_invalid"] == 0.5


def test_per_kind_has_the_healthy_control_as_its_own_column():
    a = aggregate([rec(0, kind="B_OK"), rec(1, kind="A2", v2=False)])
    assert set(a["per_kind"]) == {"B_OK", "A2"} and a["per_kind"]["B_OK"]["v2"] == 1.0 and a["per_kind"]["A2"]["v2"] == 0.0


def test_gate_rule_thresholds():
    easy = {"esft_full": dict(n=100, v1_ne_v2=0.05, preserving_given_label_correct=0.9), "g": dict(n=100, v1_ne_v2=0.30, preserving_given_label_correct=0.5)}
    assert verdict(easy)["go"] is False and verdict(easy)["b_pass"] is False and verdict(easy)["a_pass"] is True
    ok = {"esft_full": dict(n=100, v1_ne_v2=0.12, preserving_given_label_correct=0.6), "base": dict(n=100, v1_ne_v2=0.0, preserving_given_label_correct=0.1)}
    v = verdict(ok)
    assert v["go"] is True and abs(v["a_pooled_v1_ne_v2"] - 0.12) < 1e-9      # base is not a trained policy: excluded from the pool
