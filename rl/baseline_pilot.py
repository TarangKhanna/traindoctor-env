"""Step-4 baseline pilot: untrained Qwen3-0.6B-4bit on stratified DEV episodes, G completions each.
Official score = strict gate (tiera.reward). DIAGNOSTIC score = same reward after stripping one leading ```json fence
and one trailing ``` (done here, outside tiera.py; never an official number). Not a learning claim."""
from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path


def unfence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*\n?", "", t, count=1)
    return re.sub(r"\n?\s*```\s*$", "", t, count=1).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
    from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
    from rl.first_update import strip_special
    from traindoctor_env.pilot import prepare, stratified_seeds
    from traindoctor_env.tiera import parse_answer, reward, training_prompt

    cache = out / "episodes.pkl"
    if cache.exists():
        eps = pickle.loads(cache.read_bytes())
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as wd:
            eps, skipped = [], []
            from traindoctor_env.pilot import DEV_SEEDS
            from traindoctor_env.scenario import make_scenario
            for s in stratified_seeds(a.episodes):
                kind = make_scenario(s).kind
                cand = [s] + [x for x in DEV_SEEDS if x > s and make_scenario(x).kind == kind]
                for x in cand:  # a prepare() crash (launcher IndexError) skips that seed; next DEV seed of same kind
                    try:
                        eps.append(prepare(x, Path(wd))); break
                    except Exception as e:  # noqa: BLE001
                        skipped.append((x, kind, repr(e)[:80])); print("SKIPPED", skipped[-1], flush=True)
            (out / "skipped_seeds.json").write_text(json.dumps(skipped))
        cache.write_bytes(pickle.dumps(eps))
    prompts = [training_prompt(e[1]) for e in eps]
    model, tok, _ = load_policy_with_lora("mlx-community/Qwen3-0.6B-4bit",
                                          LoRAConfig(rank=8, scale=20.0, dropout=0.0, grad_checkpoint=False))
    cfg = SamplingConfig(max_tokens=a.max_tokens, temperature=1.0, top_p=0.95)
    t0 = time.perf_counter()
    comps = generate_group_rollouts(model=model, tokenizer=tok, prompts=prompts, group_size=a.group, config=cfg,
                                    seed=a.seed, use_chat_template=False)
    secs = time.perf_counter() - t0
    rows = []
    for c in comps:
        truth, bundle, menu, table = eps[c.prompt_index]
        txt = strip_special(c.text); ids = {m.id for m in menu}
        r = dict(ep=c.prompt_index, seed=truth.seed if hasattr(truth, "seed") else None, kind=getattr(truth, "kind", None),
                 ntok=len(c.completion_tokens), text=txt)
        for name, s in (("strict", txt), ("fence", unfence(txt))):
            r0, r2 = reward(s, truth, bundle, menu, table, "V0"), reward(s, truth, bundle, menu, table, "V2")
            ans = parse_answer(s, ids)
            r[name] = dict(valid=r2.valid, unsafe=r2.unsafe, v0=r0.reward, v2=r2.reward,
                           label=ans.label if ans else None, repair=ans.repair if ans else None)
        rows.append(r)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    G, S = a.group, {}
    for name in ("strict", "fence"):
        def grp(k):
            return [[r[name][k] for r in rows[g * G:(g + 1) * G]] for g in range(len(rows) // G)]
        v0, v2 = grp("v0"), grp("v2")
        flat = lambda x: [y for g in x for y in g]  # noqa: E731
        S[name] = dict(
            valid=f"{sum(flat(grp('valid')))}/{len(rows)}", unsafe=sum(flat(grp("unsafe"))),
            v0_mean=st.mean(flat(v0)), v0_std=st.pstdev(flat(v0)), v2_mean=st.mean(flat(v2)), v2_std=st.pstdev(flat(v2)),
            groups_spread_v0=sum(max(g) != min(g) for g in v0), groups_spread_v2=sum(max(g) != min(g) for g in v2),
            groups=len(v0), per_ep_valid=[sum(grp("valid")[g]) for g in range(len(v0))],
            per_ep_v0=[sum(x) for x in v0], per_ep_v2=[sum(x) for x in v2],
            labels=Counter(str(r[name]["label"]) for r in rows if r[name]["valid"]).most_common(),
            repairs=Counter(str(r[name]["repair"]) for r in rows if r[name]["valid"]).most_common())
    S["tokens_total"] = sum(r["ntok"] for r in rows); S["tokens_mean"] = S["tokens_total"] / len(rows)
    S["gen_seconds"] = secs; S["note"] = "fence = DIAGNOSTIC only; strict is the official score. Run host idle per coordinator."
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
