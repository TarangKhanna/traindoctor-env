"""Step-3 feasibility: memory/time of the GRPO update path on real ~1.9k-token prompts, G=4, checkpointing on.
Uses lr=0 so the throwaway optimizer step changes nothing; no learning, no adapter saved."""
from __future__ import annotations

import argparse, json, os, pickle, resource, sys, threading, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rl.model_screen import swap_mb
    base = swap_mb(); beat = [time.time()]

    def guard():
        while True:
            time.sleep(10)
            if swap_mb() - base > 1500 or time.time() - beat[0] > 600:
                print("GUARD ABORT", swap_mb(), flush=True); os._exit(3)
    threading.Thread(target=guard, daemon=True).start()
    import mlx.core as mx, mlx.optimizers as optim
    from mlxrl.algo.grpo import algorithm_by_name
    from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
    from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
    from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
    from mlxrl.train.grpo import batch_from_rollouts, optimizer_step
    from rl import lp_patch
    from rl.render import render
    lp_patch.install("slice")
    eps = pickle.loads(Path("out/pilot16/episodes.pkl").read_bytes())
    model, tok, _ = load_policy_with_lora(a.model, LoRAConfig(rank=8, scale=20.0, dropout=0.0, grad_checkpoint=True))
    pad = pad_token_id_from_tokenizer(tok); algo = algorithm_by_name("grpo")
    comps = generate_group_rollouts(model=model, tokenizer=tok, prompts=[render(a.model, tok, eps[0][1])], group_size=4,
                                    config=SamplingConfig(max_tokens=256, temperature=1.0, top_p=0.95), seed=0, use_chat_template=False)
    rewards = [0.0, 1.0, 0.0, 1.0]  # synthetic spread: memory probe only, not a reward claim
    R = dict(model=a.model, prompt_len=len(comps[0].prompt_tokens), swap_base_mb=base, runs=[])
    for mb in (1, 2):
        beat[0] = time.time(); mx.reset_peak_memory(); t0 = time.perf_counter()
        batch = batch_from_rollouts(model=model, completions=comps, rewards=rewards, group_size=4, pad_token_id=pad,
                                    algorithm=algo, use_checkpoint=True, compute_reference=False)
        mx.eval(batch.old_policy_logprobs); t1 = time.perf_counter()
        optimizer_step(model=model, optimizer=optim.Adam(learning_rate=0.0), batch=batch, beta=0.0, pad_token_id=pad,
                       algorithm=algo, use_checkpoint=True, micro_batch_size=mb)
        mx.eval(model.parameters()); mx.synchronize(); t2 = time.perf_counter()
        step = t2 - t0
        R["runs"].append(dict(micro_batch=mb, logprob_prep_s=t1 - t0, update_s=t2 - t1, step_s=step,
                              peak_mlx_gb=mx.get_peak_memory() / 1e9, maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                              swap_delta_mb=swap_mb() - base, train_updates_per_hour_excl_rollout=3600 / step))
        print(json.dumps(R["runs"][-1]), flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
