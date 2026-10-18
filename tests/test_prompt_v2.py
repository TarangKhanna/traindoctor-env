"""Prompt v2 (neutral framing + NO_FAULT definition). Rules: (1) v1 stays byte-identical to the original frame so old
results reproduce; (2) v2 never presupposes a failure and defines NO_FAULT; (3) v2 changes only the header, nothing a
verifier or outcome table depends on (records, config, repair menu, FORMAT); (4) rl.render is the only path and a data
dir's `prompt_version` marker selects the version."""
import pickle
from pathlib import Path

import pytest

from modal_train import core
from rl.render import render
from traindoctor_env.bundle import Bundle
from traindoctor_env.tiera import FORMAT, apply_qwen3, messages, training_prompt

V1_TEXT = ("A distributed training job failed or misbehaved. Answer with: label, the ids of the records that support it "
           "(at most 8), the id of the repair to apply, and whether you claim it fixes the job.\n"
           "Labels: A, NO_FAULT, B\n\n== job/config.yaml ==\nx: 1\n\n== records ==\nE1 rec\n\n== repairs ==\nR1:\ndiff")
B = Bundle(0, (), {}, (), V1_TEXT, 1)
QWEN = "mlx-community/Qwen3-4B-4bit"


def test_v1_is_unchanged_default():
    assert training_prompt(B) == training_prompt(B, "v1") == apply_qwen3([{"role": "user", "content": V1_TEXT + "\n\n" + FORMAT}])
    assert render(QWEN, None, B) == render(QWEN, None, B, "v1") == training_prompt(B)


def test_v2_is_neutral_defines_no_fault_and_changes_only_the_header():
    p = training_prompt(B, "v2")
    assert "failed or misbehaved" not in p and "Determine whether this job has a fault." in p
    assert "NO_FAULT: a healthy run" in p and "relaunch without change" in p
    old, new = V1_TEXT.split("\n"), messages(B, "v2")[0]["content"].split("\n")
    assert new[0].startswith("Determine whether this job has a fault. Answer with:") and new[0].split("Answer with:")[1] == old[0].split("Answer with:")[1]
    assert new[1] == old[1] and new[3:] == old[2:] + ["", FORMAT]    # label list, records, config, menu, FORMAT untouched


def test_render_routes_version_and_rejects_unknown():
    assert render(QWEN, None, B, "v2") == training_prompt(B, "v2") != render(QWEN, None, B, "v1")
    assert core.render_prompt("Qwen/Qwen3-4B", None, B, "v2") == training_prompt(B, "v2")
    with pytest.raises(AssertionError):
        render(QWEN, None, B, "v3")


def test_data_dir_marker_selects_version(tmp_path):
    assert core.data_prompt_version(tmp_path) == "v1"          # old caches have no marker
    (tmp_path / "prompt_version").write_text("v2\n")
    assert core.data_prompt_version(tmp_path) == "v2"


def test_cached_v2_dir_is_the_same_episodes_with_the_v2_marker():
    root = Path(core.__file__).parent
    new = root / "cache_v2prompt"
    if not (new / "train_eps.pkl").exists():
        pytest.skip("cache_v2prompt not built (python -m modal_train.prompt_v2_cache)")
    assert core.data_prompt_version(new) == "v2"
    for n in ("train_eps.pkl", "dev_eps.pkl", "train_oracle.jsonl", "dev_oracle.jsonl"):
        assert (new / n).read_bytes() == (root / "cache" / n).read_bytes(), n
    for e in pickle.loads((new / "train_eps.pkl").read_bytes()):
        assert render(QWEN, None, e[1], "v2").count("NO_FAULT") == e[1].text.count("NO_FAULT") + 1
