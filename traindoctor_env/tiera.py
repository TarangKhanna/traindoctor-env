"""Tier A single-turn prompt (Qwen3 chat template, thinking off), answer parsing, and the V0/V2 reward as pure
functions. The training loop and the eval harness both build their prompt through `messages`/`apply_qwen3`."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from jobs.build import ALL_LABELS
from traindoctor_env.bundle import Bundle, Candidate, answer_verdict
from traindoctor_env.reexec import FixOutcome
from traindoctor_env.scenario import Truth
from traindoctor_env.verifiers import EVIDENCE_CAP

FORMAT = ('Output the single JSON object directly, with no markdown code fence and no text before or after it: {"label": <label>, "evidence": [<record ids>], '
          '"repair": <repair id>, "claims_fixed": <true|false>}')


PROMPT_VERSIONS = ("v1", "v2")
_V1_INTRO = "A distributed training job failed or misbehaved. Answer with:"
_V2_INTRO = "Determine whether this job has a fault. Answer with:"
NO_FAULT_DEF = ("NO_FAULT: a healthy run. There is no fault, and the correct repair is to relaunch without change.")


def reframe_v2(text: str) -> str:
    """v2 prompt (2026-10-18): v1 told the model the job "failed or misbehaved" and never described a healthy job, so the
    healthy control (NO_FAULT) was unlearnable. v2 frames the task neutrally and defines NO_FAULT right after the label
    list. Derived from the v1 bundle text (first line = intro, second = "Labels: ..."); the rest is untouched, so cached
    bundles and their outcome tables stay valid."""
    head, labels, rest = text.split("\n", 2)
    assert head.startswith(_V1_INTRO) and labels.startswith("Labels: "), "unexpected bundle header"
    return "\n".join([_V2_INTRO + head[len(_V1_INTRO):], labels, NO_FAULT_DEF, rest])


def prompt_text(bundle: Bundle, version: str = "v1") -> str:
    assert version in PROMPT_VERSIONS, version
    return bundle.text if version == "v1" else reframe_v2(bundle.text)


def messages(bundle: Bundle, version: str = "v1") -> list[dict]:
    """What the eval harness sends: one user turn."""
    return [{"role": "user", "content": prompt_text(bundle, version) + "\n\n" + FORMAT}]


def apply_qwen3(msgs: list[dict], enable_thinking: bool = False) -> str:
    """The Qwen3 chat template for user/assistant turns, with the generation prompt. With thinking off the
    template appends an empty think block to the generation prompt."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"
    return out if enable_thinking else out + "<think>\n\n</think>\n\n"


def training_prompt(bundle: Bundle, version: str = "v1") -> str:
    return apply_qwen3(messages(bundle, version))


@dataclass(frozen=True)
class Answer:
    label: str
    evidence: tuple
    repair: str
    claims_fixed: bool


def _object(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _constant(name):
    raise ValueError(name)


def parse_answer(completion: str, menu_ids: set[str]) -> Answer | None:
    """The format gate: None unless the completion is one JSON object of the right shape with a known label and a
    known repair id and at most EVIDENCE_CAP string evidence ids."""
    text = re.sub(r"^\s*<think>\s*</think>", "", completion).strip()
    try:
        o = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, RecursionError):
        return None
    if not (isinstance(o, dict) and set(o) == {"label", "evidence", "repair", "claims_fixed"}):
        return None
    ev = o["evidence"]
    if (not isinstance(o["label"], str) or not isinstance(o["repair"], str) or o["label"] not in ALL_LABELS
            or o["repair"] not in menu_ids or not isinstance(o["claims_fixed"], bool)
            or not isinstance(ev, list) or not all(isinstance(i, str) for i in ev) or len(set(ev)) > EVIDENCE_CAP):
        return None
    return Answer(o["label"], tuple(ev), o["repair"], o["claims_fixed"])


REWARD_VERSIONS = ("V0", "V1", "V2")


@dataclass(frozen=True)
class Reward:
    reward: float
    valid: bool          # passed the format gate
    unsafe: bool         # chose a destructive repair
    v0: bool = False
    v1: bool = False
    v2: bool = False


def reward(completion: str, truth: Truth, bundle: Bundle, menu: list[Candidate], table: dict[str, FixOutcome],
           version: str = "V2", format_check: str | None = None) -> Reward:
    """Invalid output scores 0.0 and is flagged: the format gate is a gate, not a term. Otherwise the reward is
    1.0 iff the chosen rung (V0 label correct; V1 label correct and evidence grounded; V2 verified, evidence-grounded,
    safe repair) holds, else 0.0. V1 is an opt-in training reward added for the reward-comparison protocol; V0 and V2
    are unchanged. An explicit format_check must match the chosen outcome's provenance; otherwise the table's
    own recorded check governs."""
    assert version in REWARD_VERSIONS, version
    a = parse_answer(completion, {c.id for c in menu})
    if a is None:
        return Reward(0.0, False, False)
    v = answer_verdict(truth, bundle, menu, table, a.label, list(a.evidence), a.repair, a.claims_fixed,
                       format_check=format_check)
    unsafe = next(c for c in menu if c.id == a.repair).destructive
    return Reward(float({"V0": v.v0, "V1": v.v1, "V2": v.v2}[version]), True, unsafe, v.v0, v.v1, v.v2)
