"""Tier A prompt length in TRUE Qwen3 tokens (hard cap 3,072). Rules: (1) est_tokens is an upper bound
of the real tokenizer's count on text of the shapes a bundle contains; (2) the chat-template + answer-format frame
costs no more than PROMPT_OVERHEAD; (3) every executed bundle's real prompt is <= CAP_TOKENS. Rules 1-2 skip with a
reason when the tokenizer is not in the offline HF cache; rule 3 also needs executed snapshots (Gloo)."""
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from traindoctor_env.bundle import CAP_TOKENS, PROMPT_OVERHEAD, est_tokens, make_bundle, make_menu, padded_config
from traindoctor_env.scenario import make_scenario
from traindoctor_env.tiera import training_prompt

from tests.test_bundle_leak import fake_snapshot


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-4bit", local_files_only=True)
    except Exception as e:                                     # not cached on this machine
        pytest.skip(f"Qwen3 tokenizer unavailable offline: {type(e).__name__}")


def _records(seed: int) -> str:
    r = random.Random(seed)
    rid = lambda: "E" + "".join(r.choice("0123456789abcdef") for _ in range(6))
    out = []
    for _ in range(r.randrange(10, 25)):
        run, rk, st = f"run{r.randrange(3)}", r.randrange(4), r.randrange(12)
        out.append(r.choice([
            f"{rid()} [{run}] rank {rk} step={st} samples={[r.randrange(1000) for _ in range(r.choice([4, 8]))]}",
            f"{rid()} [{run}] rank {rk} step={st} seq={r.randrange(40)} all_reduce bucket={r.randrange(3)} numel={r.randrange(5000)} done",
            f"{rid()} [{run}] rank {rk} exited exit=0 last_step={st}",
            f"{rid()} [{run}] resume_from=ckpt-{st} started_at_step={st} -> failed last_step={st} error=COLLECTIVE_TIMEOUT",
            f"{rid()} [{run}] rank {rk} INFO step {st} loss={r.random():.4f} lr=0.1",
            f"{rid()} ckpt-{st} step={st} world_size=4"]))
    return "\n".join(out)


def test_estimate_bounds_the_real_token_count_from_above(tok):
    texts = [_records(s) for s in range(100)]
    for s in range(10000, 10030):
        with tempfile.TemporaryDirectory() as t:
            snap, truth = fake_snapshot(make_scenario(s), Path(t) / "jd")
            menu = make_menu(snap, truth)
            texts.append("\n".join(f"{c.id}:\n{c.diff(snap.files)}" for c in menu))
            texts.append(padded_config(snap.files["job/config.yaml"]))
    worst = max(len(tok.encode(x)) / est_tokens(x) for x in texts)
    assert worst <= 1.0, worst


def test_prompt_frame_costs_no_more_than_the_overhead(tok):
    assert len(tok.encode(training_prompt(SimpleNamespace(text="")))) <= PROMPT_OVERHEAD


@pytest.mark.parametrize("name", ["A1_cfg", "A1_src", "A2", "B_SKIP", "B_DUP", "B_OK"])
def test_executed_bundle_prompt_is_within_the_cap_in_true_tokens(name, tok, snapshot, menu_table):
    menu, truth, _ = menu_table(name)
    bundle = make_bundle(snapshot(name), menu, truth)
    assert len(tok.encode(training_prompt(bundle))) <= CAP_TOKENS


SYNTH_SEEDS = range(10000, 10120)    # DEV seeds: make_scenario draws kind and 3-4 ranks, so Task B at both world sizes is covered


def test_synthetic_bundles_fit_the_cap_in_true_tokens_for_task_b_at_3_and_4_ranks(tok):
    """No Gloo here, so records are synthesized in the executed format (tests/synth_snapshot.py; pessimistic: every rank of
    the failed run records samples through the kill step). Rule: the certificate is never dropped (the Task B certificate does not fit in 2048 tokens), so
    the cap must admit it for every variant: no BundleTooLarge, estimate and real count both <= CAP_TOKENS."""
    from tests.synth_snapshot import synth_snapshot
    seen = set()
    for s in SYNTH_SEEDS:
        sc = make_scenario(s)
        with tempfile.TemporaryDirectory() as t:
            snap, truth = synth_snapshot(sc, Path(t) / "jd")
            menu = make_menu(snap, truth)
            bundle = make_bundle(snap, menu, truth)              # raises BundleTooLarge when the certificate does not fit
        n = len(tok.encode(training_prompt(bundle)))
        assert n <= bundle.tokens <= CAP_TOKENS, (s, n, bundle.tokens)
        seen.add((sc.kind, sc.world_size))
    assert {(k, w) for k in ("B_SKIP", "B_DUP", "B_OK") for w in (3, 4)} <= seen
