"""SFT warm-start: completion-only cross-entropy on environment-oracle gold answers (out/sft/train.jsonl), LoRA via mlxrl.
Supervision is the environment's reference solution, not LLM output. Not a learning claim about RL."""
from __future__ import annotations

import argparse, json, os, random, resource, sys, threading, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit"); p.add_argument("--data", default="out/sft/train.jsonl")
    p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=2); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=4); p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rl.model_screen import swap_mb
    base, beat = swap_mb(), [time.time()]

    def guard():
        while True:
            time.sleep(10)
            if swap_mb() - base > 1500 or time.time() - beat[0] > 600:
                print("GUARD ABORT", flush=True); os._exit(3)
    threading.Thread(target=guard, daemon=True).start()
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_map, tree_flatten
    from mlxrl.policy.logprobs import completion_logprobs, pad_token_id_from_tokenizer
    from mlxrl.policy.model import LoRAConfig, load_policy_with_lora
    from rl import lp_patch
    lp_patch.install("slice")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data)]
    if a.limit: rows = rows[: a.limit]
    model, tok, _ = load_policy_with_lora(a.model, LoRAConfig(rank=8, scale=20.0, dropout=0.0, grad_checkpoint=True))
    pad = pad_token_id_from_tokenizer(tok)
    ex = [(tok.encode(r["prompt"]), tok.encode(r["target"] + "<|im_end|>")) for r in rows]
    opt = optim.Adam(learning_rate=a.lr)

    def loss_fn(m, pt, ct):
        lp = completion_logprobs(m, [pt], [ct], pad, use_checkpoint=True)
        return -(lp.logprobs.sum() / lp.mask.sum())
    vg = nn.value_and_grad(model, loss_fn)
    log, t0, step = [], time.perf_counter(), 0
    rng = random.Random(a.seed)
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(ex))); rng.shuffle(order)
        for i in range(0, len(order) - a.accum + 1, a.accum):
            gsum, ls = None, []
            for j in order[i:i + a.accum]:
                l, g = vg(model, ex[j][0], ex[j][1]); mx.eval(l, g)
                g = tree_map(lambda x: x / a.accum, g)
                gsum = g if gsum is None else tree_map(lambda x, y: x + y, gsum, g)
                ls.append(l.item()); beat[0] = time.time()
            opt.update(model, gsum); mx.eval(model.parameters(), opt.state); step += 1
            log.append(dict(epoch=ep, step=step, loss=sum(ls) / len(ls), t=time.perf_counter() - t0))
            print(json.dumps(log[-1]), flush=True)
    mx.save_safetensors(str(out / "adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "sft_log.json").write_text(json.dumps(dict(args=vars(a), n=len(ex), log=log, maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                                                      swap_delta_mb=swap_mb() - base, wall_s=time.perf_counter() - t0), indent=1))


if __name__ == "__main__":
    main()
