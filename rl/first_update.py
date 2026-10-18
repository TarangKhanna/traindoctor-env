"""First verified local RL weight update (plumbing, Qwen3-0.6B-4bit, Tier A single-turn).

Loop: Tier A prompts for DEV episodes -> mlxrl rollouts -> traindoctor_env.tiera.reward (V0 and V2) ->
batch_from_rollouts -> ONE optimizer_step on the groups with reward spread -> save adapter -> reload into a fresh
model -> logprobs changed vs base and reload reproduces post-update logprobs.
Not a learning claim. mlxrl is wrapped, not edited; completion-only logits come from rl/lp_patch.py ("slice").
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path


def mixed_groups(rewards, group_size: int) -> list[int]:
    """Indices of groups whose rewards are not all equal. Only these carry a nonzero group-relative advantage."""
    assert len(rewards) % group_size == 0
    out = []
    for g in range(len(rewards) // group_size):
        grp = rewards[g * group_size:(g + 1) * group_size]
        if max(grp) != min(grp):
            out.append(g)
    return out


SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>")


def strip_special(text: str) -> str:
    """Drop trailing stop-token artifacts that the decoder leaves in the completion text."""
    t = text.rstrip()
    while True:
        for s in SPECIAL_TOKENS:
            if t.endswith(s):
                t = t[: -len(s)].rstrip()
                break
        else:
            return t


def classify(text: str, ntok: int, cap: int, valid: bool, v0: bool, v2: bool) -> str:
    """Why a completion earned no reward (or 'ok'). Order: truncation, then gate failures, then wrong answers."""
    if valid:
        return "ok" if v2 else ("wrong_repair_or_evidence" if v0 else "wrong_label")
    if ntok >= cap:
        return "truncated"
    t = text.strip()
    if t.startswith("```"):
        return "fence"
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        return "no_json"
    if i > 0 or j < len(t) - 1:
        return "extra_prose"
    try:
        o = json.loads(t)
    except ValueError:
        return "malformed_json"
    return "bad_schema_or_unknown_id" if isinstance(o, dict) else "other"


def logprob_gap(a, b) -> tuple[float, float]:
    """(max, mean) absolute per-token difference between two lists of per-token logprob lists."""
    d = [abs(x - y) for ra, rb in zip(a, b, strict=True) for x, y in zip(ra, rb, strict=True)]
    return (max(d), sum(d) / len(d)) if d else (0.0, 0.0)


def reload_reproduces(post, reloaded, atol: float) -> bool:
    return logprob_gap(post, reloaded)[0] <= atol


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def host_state():
    return dict(loadavg=os.getloadavg(), therm=_sh(["pmset", "-g", "therm"]), batt=_sh(["pmset", "-g", "batt"]),
                swap=_sh(["sysctl", "-n", "vm.swapusage"]),
                competing=_sh(["pgrep", "-fl", "mlx_lm|sglang|mlxrl|pytest|torch|first_update"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--train-reward", choices=["V0", "V2"], default="V2")
    p.add_argument("--rollout-seeds", type=int, default=3, help="re-sample all groups up to this many times looking for reward spread")
    p.add_argument("--plumbing-fallback", action="store_true",
                   help="if no real-reward spread: ONE labelled plumbing step on a format-validity reward (not V0/V2)")
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlxrl.algo.grpo import algorithm_by_name
    from mlxrl.policy.logprobs import completion_logprobs, pad_token_id_from_tokenizer
    from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
    from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
    from mlxrl.train.grpo import batch_from_rollouts, optimizer_step

    from rl import lp_patch
    from traindoctor_env.pilot import prepare, stratified_seeds
    from traindoctor_env.tiera import parse_answer, reward, training_prompt

    R = dict(args=vars(a), host_start=host_state())
    t_all = time.perf_counter()

    # 1. episodes: real re-execution + outcome table (cached so reruns skip the ~30 s/episode execution)
    cache = out / "episodes.pkl"  # pickle written and read only by this script in its own output dir
    seeds = stratified_seeds(a.episodes)
    t0 = time.perf_counter()
    if cache.exists():
        eps = pickle.loads(cache.read_bytes())
        R["outcome_table"] = "real re-execution (pilot.prepare), loaded from cache written by an earlier run of this script"
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as wd:
            eps = [prepare(s, Path(wd)) for s in seeds]
        cache.write_bytes(pickle.dumps(eps))
        R["outcome_table"] = "real re-execution (pilot.prepare), Gloo loopback worked in this shell"
    R["prepare_s"] = time.perf_counter() - t0
    R["episodes"] = [dict(seed=e[0].seed, kind=e[0].kind, bundle_tokens=e[1].tokens) for e in eps]
    prompts = [training_prompt(e[1]) for e in eps]

    # 2. model + completion-only logits
    lp_patch.install("slice")
    t0 = time.perf_counter()
    model, tok, _ = load_policy_with_lora(a.model, LoRAConfig(rank=8, scale=20.0, dropout=0.0, grad_checkpoint=True))
    mx.eval(model.parameters())
    R["load_s"] = time.perf_counter() - t0
    pad = pad_token_id_from_tokenizer(tok)
    algo = algorithm_by_name("grpo")
    G = a.group

    def seq_logprobs(m, comps):
        """Per-token logprobs of fixed completions, one row at a time (eval mode, identical call for every compare)."""
        m.eval()
        rows = []
        for c in comps:
            lp = completion_logprobs(m, [c.prompt_tokens], [c.completion_tokens], pad)
            mx.eval(lp.logprobs)
            rows.append(lp.logprobs[0, : len(c.completion_tokens)].astype(mx.float32).tolist())
        return rows

    # 3. rollouts + rewards (V0 and V2 for every completion); resample until some group has spread
    sampling = SamplingConfig(max_tokens=a.max_tokens, temperature=a.temperature, top_p=0.95)
    attempts = []
    chosen = None
    for k in range(a.rollout_seeds):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        comps = generate_group_rollouts(model=model, tokenizer=tok, prompts=prompts, group_size=G, config=sampling,
                                        seed=a.seed + k, use_chat_template=False)
        mx.synchronize()
        dt = time.perf_counter() - t0
        rows, rv0, rv2 = [], [], []
        for c in comps:
            truth, bundle, menu, table = eps[c.prompt_index]
            ctext = strip_special(c.text)
            r0 = reward(ctext, truth, bundle, menu, table, "V0")
            r2 = reward(ctext, truth, bundle, menu, table, "V2")
            rv0.append(r0.reward); rv2.append(r2.reward)
            rows.append(dict(prompt=c.prompt_index, ntok=len(c.completion_tokens), valid=r2.valid, unsafe=r2.unsafe,
                             v0=r0.reward, v2=r2.reward, text=ctext, raw_text=c.text,
                             reason=classify(ctext, len(c.completion_tokens), a.max_tokens, r2.valid, r0.v0, r2.v2)))
        ntok = sum(r["ntok"] for r in rows)
        info = dict(seed=a.seed + k, rollout_s=dt, gen_tokens=ntok, gen_tok_s=ntok / dt,
                    peak_gb=mx.get_peak_memory() / 1e9, prompt_len=len(comps[0].prompt_tokens),
                    valid=sum(r["valid"] for r in rows), n=len(rows), mean_v0=sum(rv0) / len(rv0),
                    mean_v2=sum(rv2) / len(rv2), mixed_v0=mixed_groups(rv0, G), mixed_v2=mixed_groups(rv2, G),
                    reasons={x: sum(r["reason"] == x for r in rows) for x in sorted({r["reason"] for r in rows})},
                    loadavg=os.getloadavg(), completions=rows)
        attempts.append(info)
        print(json.dumps({k_: v for k_, v in info.items() if k_ != "completions"}), flush=True)
        tr = rv0 if a.train_reward == "V0" else rv2
        if mixed_groups(tr, G):
            chosen = (comps, tr, f"real {a.train_reward} reward")
            break
        last = (comps, rows)
    R["rollout_attempts"] = attempts
    if chosen is None and a.plumbing_fallback:
        comps, rows = last
        chosen = (comps, [1.0 - len(c.completion_tokens) / a.max_tokens for c in comps],
                  "PLUMBING ONLY: shorter-completion reward (not V0/V2; no semantic claim)")
        if not mixed_groups(chosen[1], G):
            chosen = None
    if chosen is None:
        R["update"] = "NOT PERFORMED: no group had reward spread"
        R["host_end"] = host_state()
        (out / "result.json").write_text(json.dumps(R, indent=1, default=str))
        print("NO MIXED GROUP; recorded", flush=True)
        return
    comps, rewards, reward_kind = chosen
    sel = mixed_groups(rewards, G)
    sel_comps = [c for g in sel for c in comps[g * G:(g + 1) * G]]
    sel_rewards = [x for g in sel for x in rewards[g * G:(g + 1) * G]]
    R["update_reward"] = reward_kind
    R["trained_groups"] = sel
    R["trained_group_rewards"] = [rewards[g * G:(g + 1) * G] for g in sel]
    R["skipped_zero_spread_groups"] = [g for g in range(len(comps) // G) if g not in sel]

    # 4. base logprobs (LoRA B == 0, so this is the base model), then ONE update
    lp_base = seq_logprobs(model, sel_comps)
    trainable = lambda: {k: v + 0 for k, v in tree_flatten(model.trainable_parameters())}  # noqa: E731
    p0 = trainable(); mx.eval(list(p0.values()))
    opt = optim.Adam(learning_rate=a.lr)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    batch = batch_from_rollouts(model=model, completions=sel_comps, rewards=sel_rewards, group_size=G, pad_token_id=pad,
                                algorithm=algo, use_checkpoint=True, compute_reference=False)
    mx.eval(batch.old_policy_logprobs); t1 = time.perf_counter()
    m = optimizer_step(model=model, optimizer=opt, batch=batch, beta=0.0, pad_token_id=pad, algorithm=algo,
                       use_checkpoint=True, micro_batch_size=a.micro_batch)
    mx.eval(model.parameters(), opt.state); mx.synchronize(); t2 = time.perf_counter()
    p1 = trainable(); mx.eval(list(p1.values()))
    changed = [k for k in p0 if float(mx.max(mx.abs(p1[k].astype(mx.float32) - p0[k].astype(mx.float32))).item()) > 0]
    R["update"] = dict(logprob_prep_s=t1 - t0, step_s=t2 - t1, peak_gb=mx.get_peak_memory() / 1e9,
                       maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                       loss=m.loss, mean_reward=m.mean_reward, clip_fraction=m.clip_fraction, kl=m.kl,
                       advantages=[round(x, 4) for x in batch.advantages.tolist()],
                       lora_tensors_changed=len(changed), lora_tensors_total=len(p0),
                       lora_max_abs_param_diff=max(float(mx.max(mx.abs(p1[k].astype(mx.float32) - p0[k].astype(mx.float32))).item()) for k in p0),
                       completions_in_step=len(sel_comps))

    # 5. post-update logprobs, save adapter, reload into a FRESH model
    lp_post = seq_logprobs(model, sel_comps)
    ad = out / "adapter.safetensors"
    mx.save_safetensors(str(ad), dict(tree_flatten(model.trainable_parameters())))
    fresh, _, _ = load_policy_with_lora(a.model, LoRAConfig(rank=8, scale=20.0, dropout=0.0, grad_checkpoint=True))
    lp_fresh_before = seq_logprobs(fresh, sel_comps)
    saved = mx.load(str(ad))
    fresh.load_weights(list(saved.items()), strict=False)
    fp = dict(tree_flatten(fresh.trainable_parameters()))
    mx.eval(list(fp.values()))
    assert set(fp) == set(saved), "adapter keys do not match the fresh model's LoRA params"
    param_exact = all(bool(mx.array_equal(fp[k], saved[k]).item()) for k in saved)
    lp_reload = seq_logprobs(fresh, sel_comps)
    R["verify"] = dict(
        atol=a.atol, n_tokens=sum(len(r) for r in lp_base),
        base_vs_post_max_mean=logprob_gap(lp_base, lp_post),
        fresh_unloaded_vs_base_max_mean=logprob_gap(lp_base, lp_fresh_before),
        reload_vs_post_max_mean=logprob_gap(lp_post, lp_reload),
        reload_vs_base_max_mean=logprob_gap(lp_base, lp_reload),
        logprobs_changed=logprob_gap(lp_base, lp_post)[0] > a.atol,
        reload_reproduces_post=reload_reproduces(lp_post, lp_reload, a.atol),
        reloaded_params_bit_equal=param_exact)
    R["wall_total_s"] = time.perf_counter() - t_all
    R["host_end"] = host_state()
    (out / "result.json").write_text(json.dumps(R, indent=1, default=str))
    print("VERIFY " + json.dumps(R["verify"]), flush=True)
    print("UPDATE " + json.dumps({k: v for k, v in R["update"].items()}), flush=True)


if __name__ == "__main__":
    main()
