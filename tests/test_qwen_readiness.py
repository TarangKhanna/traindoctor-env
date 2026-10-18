"""Tier A prompt parity, the reward's gate/rungs, and the pilot harness, on executed scenarios (shared fixtures)."""
import json
from collections import Counter
from pathlib import Path
from dataclasses import replace

import pytest

from tests.conftest import NAMES
from traindoctor_env.bundle import CAP_TOKENS, Bundle, make_bundle
from traindoctor_env.oracle import reference_solution
from traindoctor_env.pilot import run_pilot, stratified_seeds
from traindoctor_env.scenario import make_scenario
from traindoctor_env.tiera import apply_qwen3, messages, reward, training_prompt

pytestmark = pytest.mark.executed


def ep(name, snapshot, menu_table):
    """Episode with a synthetic fix-outcome table (correct repairs ok, the rest not): these tests check the reward's
    lookup rules, not re-execution, which tests/test_tiera.py covers."""
    menu, truth, table = menu_table(name)
    proto = next(iter(table.values()))
    synth = {c.id: replace(proto, ok=c.role == "correct", tampered=(), sandboxed=True) for c in menu}
    return truth, make_bundle(snapshot(name), menu, truth), menu, synth


def completion(snapshot, name, ep_, role="correct", **over):
    truth, bundle, menu, _ = ep_
    _, ev, _ = reference_solution(snapshot(name), truth)
    o = dict(label=truth.label, evidence=ev, repair=next(c.id for c in menu if c.role == role), claims_fixed=True)
    return json.dumps({**o, **over})


GOLDEN = Path(__file__).parent / "golden"
# (fixture, bundle text): the fixture bytes were assembled by hand from the Tier A format spec (+ the Qwen3 template,
# the Qwen3 chat template: user turn, then "<|im_start|>assistant\n<think>\n\n</think>\n\n" with thinking off), not dumped.
GOLDEN_CASES = [
    ("tiera_prompt_plain.txt", "Incident bundle\n[R1] step 3 loss=nan\n[M1] diff: lr: 0.1 -> 0.01"),
    ("tiera_prompt_blank_unicode_trailing_nl.txt", "Line one\n\nR7: grad_norm=1.5e3 \u2014 spike\n"),
]


@pytest.mark.parametrize("fixture,text", GOLDEN_CASES)
def test_training_prompt_matches_the_committed_golden_bytes(fixture, text):
    """Rule (Tier A format parity): the model sees exactly the Qwen3 template bytes with thinking off.
    The expected bytes live in a committed fixture written from the spec, so a drift in the production renderer
    (template, think block, stripping, FORMAT wording) fails here instead of being echoed by both sides."""
    b = Bundle(0, (), {}, (), text, 1)
    want = (GOLDEN / fixture).read_bytes()
    assert training_prompt(b).encode("utf-8") == want
    assert apply_qwen3(messages(b)).encode("utf-8") == want


@pytest.mark.parametrize("name", NAMES)
def test_real_bundle_prompt_is_the_golden_frame_around_the_bundle_text(name, snapshot, menu_table):
    """Rule: for real bundles the prompt is exactly the golden frame around bundle.text (verbatim, unstripped) and
    stays within the token cap; the frame is taken from the committed fixture, not from production code."""
    _, bundle, _, _ = ep(name, snapshot, menu_table)
    frame = (GOLDEN / GOLDEN_CASES[0][0]).read_text(encoding="utf-8")
    head, tail = "<|im_start|>user\n", frame.split(GOLDEN_CASES[0][1], 1)[1]
    assert training_prompt(bundle) == head + bundle.text + tail and bundle.tokens <= CAP_TOKENS


@pytest.mark.parametrize("name", NAMES)
def test_reward_is_a_gate_then_a_rung(name, snapshot, menu_table):
    """Rules: an invalid answer is gated to 0 under both versions; V0 pays for the label alone; V2 additionally
    needs a verified safe repair, so a right label with a wrong repair pays V0 but not V2."""
    e = ep(name, snapshot, menu_table)
    args = e[0], e[1], e[2], e[3]
    good = reward(completion(snapshot, name, e), *args, "V2")
    assert (good.reward, good.valid, good.v2) == (1.0, True, True)
    for v in ("V0", "V2"):
        for bad in ("not json", completion(snapshot, name, e, repair="Rzzzzz"), completion(snapshot, name, e, label="NOPE")):
            r = reward(bad, *args, v)
            assert (r.reward, r.valid) == (0.0, False)
    wrong = next(c.role for c in e[2] if not e[3][c.id].ok)
    r0, r2 = (reward(completion(snapshot, name, e, role=wrong), *args, v) for v in ("V0", "V2"))
    assert (r0.reward, r2.reward, r2.valid) == (1.0, 0.0, True)


@pytest.mark.parametrize("name", NAMES)
def test_format_violations_score_zero_and_never_raise(name, snapshot, menu_table):
    """Rule: the prompt demands one JSON object and nothing else, so wrong field types, duplicate keys, NaN/Infinity, pathological nesting and any text around the
    object are invalid (reward 0, valid False) under both versions, and a malformed answer never raises."""
    e = ep(name, snapshot, menu_table)
    args = e[0], e[1], e[2], e[3]
    good = completion(snapshot, name, e)
    bad = [completion(snapshot, name, e, repair=[]), completion(snapshot, name, e, repair={}),
           completion(snapshot, name, e, label=["x"]), "Sure.</think>" + good, "Answer: " + good,
           "```json\n" + good + "\n```", good + "\nDone.",
           '{"claims_fixed": NaN, ' + good[1:], good[:-1] + ', "repair": "Rzzzzz"}', good.replace("true", "Infinity"),
           '{"label":' + "[" * 20000]
    for v in ("V0", "V2"):
        assert reward("<think>\n\n</think>\n\n" + good, *args, v).valid
        for b in bad:
            r = reward(b, *args, v)
            assert (r.reward, r.valid) == (0.0, False), b


def test_stratified_seeds_cover_every_kind_evenly():
    c = Counter(make_scenario(s).kind for s in stratified_seeds(15))
    assert set(c.values()) == {3} and len(c) == 5


def test_pilot_records_outcomes_and_reward_diversity(snapshot, menu_table):
    eps = [ep(n, snapshot, menu_table) for n in ("A1_cfg", "B_SKIP")]
    calls = iter(range(10**6))

    def generate(prompt):   # alternates a correct and an unparseable answer
        i = next(calls)
        return (completion(snapshot, "A1_cfg", eps[0]) if i % 2 == 0 else "?"), 7

    out = run_pilot(eps[:1], generate, attempts=4)
    assert (out["episodes"], out["attempts"], out["mixed_episodes"]) == (1, 4, 1)
    assert out["recovery"] == 0.5 and out["invalid"] == 0.5 and out["mean_tokens"] == 7 and out["unsafe"] == 0
