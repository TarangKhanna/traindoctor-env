"""Rules under test (pure pieces of rl/first_update.py).
GRPO advantages are group-relative (zero mean per group, exactly zero for an all-equal group), so only groups with
reward spread can move the policy; and a reload check must FAIL when the reloaded logprobs differ beyond tolerance."""
import mlx.core as mx
from mlxrl.algo.grpo import algorithm_by_name

from rl.first_update import logprob_gap, mixed_groups, reload_reproduces


def test_advantages_zero_mean_per_group_and_zero_for_equal_groups():
    r = mx.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv = algorithm_by_name("grpo").compute_advantages(r, group_structure=4).tolist()
    for g in range(3):
        assert abs(sum(adv[4 * g:4 * g + 4])) < 1e-5
    assert adv[0] > 0 > adv[1]
    assert adv[4:8] == [0.0] * 4 and adv[8:12] == [0.0] * 4


def test_mixed_groups_selects_exactly_the_groups_with_nonzero_advantage():
    r = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    adv = algorithm_by_name("grpo").compute_advantages(mx.array(r), group_structure=4).tolist()
    expect = [g for g in range(4) if any(abs(x) > 0 for x in adv[4 * g:4 * g + 4])]
    assert mixed_groups(r, 4) == expect == [0, 3]


def test_reload_check_passes_when_equal_and_fails_when_off():
    post = [[-0.5, -1.25], [-2.0]]
    assert reload_reproduces(post, [[-0.5, -1.25], [-2.0]], 1e-6)
    assert not reload_reproduces(post, [[-0.5, -1.25], [-2.001]], 1e-6)
    assert logprob_gap(post, [[-0.5, -1.0], [-2.0]]) == (0.25, 0.25 / 3)


def test_strip_special_removes_only_trailing_stop_tokens():
    from rl.first_update import strip_special
    assert strip_special('{"a": 1}<|im_end|>') == '{"a": 1}'
    assert strip_special('{"a": 1}\n<|im_end|><|endoftext|>') == '{"a": 1}'
    assert strip_special('<|im_end|>x') == '<|im_end|>x'
