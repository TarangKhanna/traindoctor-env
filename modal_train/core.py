"""Pure-Python glue (no Modal; torch/trl imported lazily by trainlib). Reuses rl/render.py, rl/gen_sft.py semantics,
rl/splits.py and rl/first_update.py helpers. Prompts are produced ONLY via rl.render at runtime (never copied)."""
from __future__ import annotations

import dataclasses
import json
from collections import defaultdict

from rl.first_update import classify, strip_special
from rl.render import render
from traindoctor_env.tiera import parse_answer, reward

# HF (CUDA) ids -> the id rl.render knows, so Qwen3 gets our exact thinking-off template, byte-identical to the MLX path.
RENDER_ALIAS = {"Qwen/Qwen3-0.6B": "mlx-community/Qwen3-0.6B-4bit", "Qwen/Qwen3-4B": "mlx-community/Qwen3-4B-4bit"}


def render_prompt(model_id: str, tok, bundle, version: str = "v1") -> str:
    return render(RENDER_ALIAS.get(model_id, model_id), tok, bundle, version)


def data_prompt_version(data_dir) -> str:
    """The prompt version a data dir was prepared for: its `prompt_version` marker file, else v1 (old caches)."""
    from pathlib import Path
    m = Path(data_dir) / "prompt_version"
    return m.read_text().strip() if m.exists() else "v1"


def score(completion: str, ep, version: str = "V2"):
    """ep = (truth, bundle, menu, table) as produced by traindoctor_env.pilot.prepare. Strict gate: invalid -> 0."""
    truth, bundle, menu, table = ep
    return reward(strip_special(completion), truth, bundle, menu, table, version)


def make_reward_fn(episodes: list, version: str):
    """TRL reward function; dataset column `idx` selects the episode. Rewards come from the local deterministic table."""
    def reward_fn(completions, idx, **kw):
        return [float(score(c, episodes[i], version).reward) for c, i in zip(completions, idx, strict=True)]
    reward_fn.__name__ = f"reward_{version}"
    return reward_fn


def dev_episode(ep):
    """TRAINING-ONLY, UNVERIFIED DEV reward view: a transient in-memory copy of the episode whose dev_container
    outcomes are marked sandboxed so the unchanged V2 verifier scores them. Nothing is persisted; the stored
    tables keep sandboxed=False / exec_mode=dev_container. Only the GRPO training reward may call this."""
    truth, bundle, menu, table = ep
    return truth, bundle, menu, {k: dataclasses.replace(o, sandboxed=True) if o.exec_mode == "dev_container" else o
                                 for k, o in table.items()}


def make_dev_reward_fn(episodes: list, version: str):
    """GRPO training reward over DEV (dev_container) outcome tables. Eval paths must use make_reward_fn/score."""
    def reward_fn(completions, idx, **kw):
        return [float(score(c, dev_episode(episodes[i]), version).reward) for c, i in zip(completions, idx, strict=True)]
    reward_fn.__name__ = f"dev_reward_{version}"
    return reward_fn


def summarize(rows: list[dict], cap: int) -> dict:
    """rows: one per sampled completion: {kind, text, ntok, valid, unsafe, v0, v1, v2, claims_fixed}."""
    n = len(rows) or 1
    claimed = [r for r in rows if r["valid"] and r["claims_fixed"]]
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["v2"])
    return dict(n=len(rows), valid_rate=sum(r["valid"] for r in rows) / n, v0=sum(r["v0"] for r in rows) / n,
                v1_evidence_correct=sum(r["v1"] for r in rows) / n, v2=sum(r["v2"] for r in rows) / n,
                unsafe_rate=sum(r["unsafe"] for r in rows) / n, claimed_fixed=len(claimed),
                claimed_but_unverified=sum(not r["v2"] for r in claimed),
                v2_by_kind={k: sum(v) / len(v) for k, v in by_kind.items()},
                reasons={x: sum(r["reason"] == x for r in rows) for x in sorted({r["reason"] for r in rows})},
                mean_completion_tokens=sum(r["ntok"] for r in rows) / n)


def eval_row(text: str, ntok: int, cap: int, ep, kind: str) -> dict:
    t = strip_special(text)
    r0, r2 = score(text, ep, "V0"), score(text, ep, "V2")
    a = parse_answer(t, {c.id for c in ep[2]})
    return dict(kind=kind, text=t, ntok=ntok, valid=r2.valid, unsafe=r2.unsafe, v0=r0.v0, v1=r2.v1, v2=r2.v2,
                claims_fixed=bool(a and a.claims_fixed), reason=classify(t, ntok, cap, r2.valid, r0.v0, r2.v2))


def read_jsonl(path) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


# --- CONTROLS (exploratory, modal_train only; the core env is untouched) ---
def variant_episode(ep, variant: str):
    """Eval-only prompt perturbation of a DEV episode. variant: none | shuffle | menu_only.
    shuffle: permute the repair-menu order AND replace every menu id by a fresh deterministic id (menu, outcome table and
    prompt text all remapped, so the unchanged scorer maps answers back). menu_only: blind baseline; prompt keeps the
    instructions, label list and repair menu but drops job/config.yaml and all records."""
    if variant == "none":
        return ep
    import hashlib, random
    from traindoctor_env.bundle import Bundle
    truth, bundle, menu, table = ep
    head, tail = bundle.text.split("\n== repairs ==\n")
    pairs = list(bundle.menu)
    assert head + "\n== repairs ==\n" + "\n".join(x for rid, d in pairs for x in (f"{rid}:", d)) == bundle.text, "render mismatch"
    if variant == "shuffle":
        order = list(range(len(menu)))
        random.Random(f"ctl-shuffle:{bundle.seed}").shuffle(order)
        new = {c.id: "R" + hashlib.sha256(f"ctl-id:{bundle.seed}:{c.id}".encode()).hexdigest()[:5] for c in menu}
        assert len(set(new.values())) == len(new)
        menu = [dataclasses.replace(menu[i], id=new[menu[i].id]) for i in order]
        diff = dict(pairs)
        pairs = [(new[menu_old], diff[menu_old]) for menu_old in (list(new)[i] for i in order)]
        table = {new[k]: v for k, v in table.items()}
    elif variant == "menu_only":
        head = "\n".join(head.split("\n")[:2])
    else:
        raise ValueError(variant)
    text = head + "\n== repairs ==\n" + "\n".join(x for rid, d in pairs for x in (f"{rid}:", d))
    return truth, dataclasses.replace(bundle, menu=tuple(pairs), text=text), menu, table
