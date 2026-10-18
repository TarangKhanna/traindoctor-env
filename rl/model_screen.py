"""Strict-gate screen of one model on N cached DEV episodes x G completions. Official = strict tiera gate;
fence = DIAGNOSTIC fence-tolerant score computed here. Not a learning claim. Guards: swap growth / stalled log."""
from __future__ import annotations

import argparse, json, os, pickle, resource, subprocess, sys, threading, time
import statistics as st
from collections import Counter
from pathlib import Path


def swap_mb() -> float:
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(s.split("used =")[1].split("M")[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes-pkl", default="out/pilot16/episodes.pkl")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=["menu_only", "evidence_only", "neutral"], default=None,
                   help="EXPERIMENT ONLY prompt controls: menu_only=drop config+records; evidence_only=repair contents withheld; neutral=both")
    p.add_argument("--adapter", default=None, help="LoRA safetensors to load on top of the base model")
    p.add_argument("--shuffle-labels", action="store_true", help="EXPERIMENT ONLY: permute the Labels line per episode (seeded)")
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base = swap_mb(); beat = [time.time()]

    def guard():
        while True:
            time.sleep(10)
            if swap_mb() - base > 1500 or time.time() - beat[0] > 600:
                print("GUARD ABORT swap=%.0f base=%.0f stall=%.0fs" % (swap_mb(), base, time.time() - beat[0]), flush=True)
                os._exit(3)
    threading.Thread(target=guard, daemon=True).start()

    import mlx.core as mx
    from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
    from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
    from rl.baseline_pilot import unfence
    from rl.first_update import strip_special
    from rl.render import render
    from traindoctor_env.tiera import parse_answer, reward

    eps = pickle.loads(Path(a.episodes_pkl).read_bytes())[: a.episodes]
    t0 = time.perf_counter()
    model, tok, _ = load_policy_with_lora(a.model, LoRAConfig(rank=8, scale=20.0, dropout=0.0, grad_checkpoint=False))
    if a.adapter:
        model.load_weights(list(mx.load(a.adapter).items()), strict=False); mx.eval(model.parameters())
    load_s = time.perf_counter() - t0; beat[0] = time.time()
    if a.shuffle_labels:
        import dataclasses, random
        from jobs.build import ALL_LABELS
        old = "Labels: " + ", ".join(ALL_LABELS)
        def shuf(b):
            l = list(ALL_LABELS); random.Random(f"labels:{b.seed}").shuffle(l)
            assert old in b.text
            return dataclasses.replace(b, text=b.text.replace(old, "Labels: " + ", ".join(l)))
        eps = [(e[0], shuf(e[1]), e[2], e[3]) for e in eps]
    if a.variant:
        import dataclasses, re
        def var(b):
            t = b.text; i, j = t.find("== job/config.yaml =="), t.find("== repairs ==")
            head, evid, rep = t[:i], t[i:j], t[j:]
            if a.variant in ("evidence_only", "neutral"):
                rep = "== repairs ==\n" + "\n".join(f"{m.group(1)}:\n(content withheld)" for m in re.finditer(r"^(R[0-9a-f]{5}):$", rep, re.M))
            if a.variant in ("menu_only", "neutral"):
                evid = ""
            return dataclasses.replace(b, text=head + evid + rep)
        eps = [(e[0], var(e[1]), e[2], e[3]) for e in eps]
    prompts = [render(a.model, tok, e[1]) for e in eps]
    (out / "prompt0.txt").write_text(prompts[0][:400] + "\n...\n" + prompts[0][-400:])
    cfg = SamplingConfig(max_tokens=a.max_tokens, temperature=1.0, top_p=0.95)
    mx.reset_peak_memory(); rows = []; gen_s = 0.0
    for i, pr in enumerate(prompts):
        t1 = time.perf_counter()
        comps = generate_group_rollouts(model=model, tokenizer=tok, prompts=[pr], group_size=a.group, config=cfg,
                                        seed=a.seed + i, use_chat_template=False)
        mx.synchronize(); dt = time.perf_counter() - t1; gen_s += dt; beat[0] = time.time()
        truth, bundle, menu, table = eps[i]; ids = {m.id for m in menu}
        for c in comps:
            txt = strip_special(c.text)
            r = dict(ep=i, kind=truth.kind, truth=truth.label, ntok=len(c.completion_tokens), prompt_len=len(c.prompt_tokens), text=txt)
            for name, s in (("strict", txt), ("fence", unfence(txt))):
                r0, r2 = reward(s, truth, bundle, menu, table, "V0"), reward(s, truth, bundle, menu, table, "V2")
                ans = parse_answer(s, ids)
                r[name] = dict(valid=r2.valid, unsafe=r2.unsafe, v0=r0.reward, v1=float(r2.v1), v2=r2.reward,
                               label=ans.label if ans else None, repair=ans.repair if ans else None,
                               repair_ok=bool(ans and table[ans.repair].ok))
            rows.append(r)
        print(f"ep {i} {truth.kind} {dt:.0f}s valid_strict={sum(r['strict']['valid'] for r in rows[-a.group:])}", flush=True)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    G = a.group; S = dict(model=a.model, adapter=a.adapter, shuffle_labels=a.shuffle_labels, variant=a.variant, episodes=len(eps), n=len(rows), swap_base_mb=base, swap_end_mb=swap_mb(),
                          load_s=load_s, gen_s=gen_s, peak_mlx_gb=mx.get_peak_memory() / 1e9,
                          maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                          tokens=sum(r["ntok"] for r in rows), prompt_len=rows[0]["prompt_len"])
    S["tok_s"] = S["tokens"] / gen_s
    for name in ("strict", "fence"):
        v = [r[name] for r in rows]
        def spread(k):
            return sum(len({x[k] for x in v[g * G:(g + 1) * G]}) > 1 for g in range(len(v) // G))
        val = [r for r in rows if r[name]["valid"]]
        S[name] = dict(valid=sum(x["valid"] for x in v), unsafe=sum(x["unsafe"] for x in v),
                       v0=st.mean(x["v0"] for x in v), v1=st.mean(x["v1"] for x in v), v2=st.mean(x["v2"] for x in v),
                       v2_std=st.pstdev(x["v2"] for x in v), spread_v0=spread("v0"), spread_v2=spread("v2"),
                       repair_ok_of_valid=sum(r[name]["repair_ok"] for r in val),
                       labels_valid=Counter(r[name]["label"] for r in val).most_common(),
                       truth_labels=Counter(r["truth"] for r in rows).most_common(), per_ep_label=[[r[name]["label"] for r in rows[g*G:(g+1)*G]] for g in range(len(rows)//G)])
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S))


if __name__ == "__main__":
    main()
