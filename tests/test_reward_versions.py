"""Reward versions (CPU, no execution): tiera.reward pays the requested rung of one verdict."""
def test_v1_reward_is_label_and_evidence_and_v0_v2_are_unchanged():
    """Rule: each reward version pays exactly its rung of the verifier ladder (V0 label; V1 label + grounded evidence; V2
    adds verified safe repair), from one verdict; an invalid answer pays nothing under every version."""
    from types import SimpleNamespace
    from unittest import mock
    from traindoctor_env import tiera
    menu = [SimpleNamespace(id="Rabc12", destructive=False)]
    good = '{"label": "NO_FAULT", "evidence": [], "repair": "Rabc12", "claims_fixed": true}'
    for v0, v1, v2 in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (0, 0, 0)]:
        with mock.patch.object(tiera, "answer_verdict", return_value=SimpleNamespace(v0=bool(v0), v1=bool(v1), v2=bool(v2))):
            got = {ver: tiera.reward(good, None, None, menu, {}, ver).reward for ver in tiera.REWARD_VERSIONS}
        assert got == {"V0": v0, "V1": v1, "V2": v2}
    assert all(tiera.reward("not json", None, None, menu, {}, ver) == tiera.Reward(0.0, False, False)
               for ver in tiera.REWARD_VERSIONS)
