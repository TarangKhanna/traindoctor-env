"""GPU-side training code (runs inside the Modal container; also importable anywhere torch/trl/peft are installed).
All TRL/PEFT specifics are isolated here. Private-API timing hooks are best-effort and flagged in the output."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path


LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")


def gpu_info() -> dict:
    import torch
    return dict(name=torch.cuda.get_device_name(0), total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else dict(name="cpu")


def peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def load_eps(data_dir: str, name: str) -> list:
    return pickle.loads((Path(data_dir) / f"{name}_eps.pkl").read_bytes())  # written by modal_train.prepare, our own file


def load_tok(model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def sft_dataset(data_dir: str, model: str, limit: int = 0):
    from modal_train.core import data_prompt_version, read_jsonl, render_prompt
    from datasets import Dataset
    pv = data_prompt_version(data_dir)
    eps, oracle = load_eps(data_dir, "train"), read_jsonl(Path(data_dir) / "train_oracle.jsonl")
    tok = load_tok(model)
    rows = [dict(prompt=render_prompt(model, tok, e[1], pv), completion=o["target"] + "<|im_end|>") for e, o in zip(eps, oracle, strict=True)]
    return Dataset.from_list(rows[:limit] if limit else rows)


def grpo_dataset(data_dir: str, model: str, limit: int = 0):
    from modal_train.core import data_prompt_version, render_prompt
    from datasets import Dataset
    pv = data_prompt_version(data_dir)
    eps = load_eps(data_dir, "train")
    tok = load_tok(model)
    rows = [dict(prompt=render_prompt(model, tok, e[1], pv), idx=i) for i, e in enumerate(eps)]
    return Dataset.from_list(rows[:limit] if limit else rows), eps


class Timing:
    """Wraps a trainer's phase methods; seconds are best-effort (private TRL attrs) and reported as such."""
    def __init__(self):
        self.t = {"rollout_and_score": 0.0, "training_step_total": 0.0, "optimizer": 0.0}
        self.steps, self.hooked = 0, []
        self._t0 = 0.0

    def wrap(self, obj, name, key):
        f = getattr(obj, name, None)
        if f is None:
            return
        def g(*a, **k):
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.t[key] += time.perf_counter() - t0
        setattr(obj, name, g); self.hooked.append(name)

    def callback(self):
        from transformers import TrainerCallback
        me = self
        class CB(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, **kw):
                me._t0 = time.perf_counter()
            def on_optimizer_step(self, args, state, control, **kw):
                me.t["optimizer"] += time.perf_counter() - me._t0
            def on_step_end(self, args, state, control, **kw):
                me.steps += 1
        return CB()

    def report(self) -> dict:
        d = dict(self.t, steps=self.steps, hooked=self.hooked)
        d["backward_and_other"] = max(0.0, self.t["training_step_total"] - self.t["rollout_and_score"])
        return d


def _model(model: str, adapter: str | None, trainable: bool):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter, is_trainable=trainable)
    return m


def sha256_dir(d) -> str:
    import hashlib
    f = Path(d) / "adapter_model.safetensors"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else ""


def delta_norm(a: str, b: str) -> dict:
    """L2 norm of (b - a) over LoRA tensors, and of a, for two saved adapters."""
    from safetensors.torch import load_file
    import torch
    x, y = load_file(f"{a}/adapter_model.safetensors"), load_file(f"{b}/adapter_model.safetensors")
    d = torch.sqrt(sum(((y[k].float() - x[k].float()) ** 2).sum() for k in x)).item()
    n = torch.sqrt(sum((x[k].float() ** 2).sum() for k in x)).item()
    return dict(delta_l2=d, base_l2=n, rel=d / max(n, 1e-12))


def run_sft(model, data_dir, out, steps=-1, epochs=2, lr=1e-4, limit=0, batch=4, accum=4, seed=0, ds=None, save_at=0, init_adapter=None) -> dict:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    ds = ds if ds is not None else sft_dataset(data_dir, model, limit)
    cfg = SFTConfig(output_dir=out, max_steps=steps, num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=batch,
                    gradient_accumulation_steps=accum, bf16=True, logging_steps=1, save_strategy="no", report_to="none",
                    completion_only_loss=True, gradient_checkpointing=True, seed=seed, max_length=4096)
    tm = Timing()
    tok = load_tok(model)
    cbs = [tm.callback()]
    if save_at:
        from transformers import TrainerCallback
        class Light(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kw):
                if state.global_step == save_at:
                    model.save_pretrained(f"{out}/adapter_light")
        cbs.append(Light())
    # init_adapter (CONTROL): continue training an existing LoRA adapter (fresh optimizer/schedule) instead of a new one.
    tr = SFTTrainer(model=_model(model, init_adapter, True), args=cfg, train_dataset=ds, processing_class=tok,
                    **({} if init_adapter else dict(peft_config=LoraConfig(**LORA))), callbacks=cbs)
    tm.wrap(tr, "training_step", "training_step_total")
    t0 = time.perf_counter()
    res = tr.train()
    wall = time.perf_counter() - t0
    tr.model.save_pretrained(f"{out}/adapter")
    seq_tok = sum(len(tok(r["prompt"] + r["completion"])["input_ids"]) for r in ds)
    return dict(stage="sft", wall_s=wall, train_loss=res.training_loss, peak_gb=peak_gb(), n=len(ds), timing=tm.report(),
                optimizer_steps=tr.state.global_step, epochs=epochs, lr=lr, batch=batch, accum=accum, save_at=save_at,
                tokens_per_epoch=seq_tok, tokens_total=seq_tok * epochs,
                init_adapter=init_adapter, adapter_sha256=sha256_dir(f"{out}/adapter"), light_sha256=sha256_dir(f"{out}/adapter_light") if save_at else None,
                loss_curve=[(x["step"], x["loss"]) for x in tr.state.log_history if "loss" in x])


def run_grpo(model, data_dir, out, arm, sft_adapter=None, steps=50, group=8, max_new=400, lr=5e-6, limit=0, prompts_per_step=2, seed=0, ds=None, reward_fn=None) -> dict:
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    if ds is None:
        from modal_train.core import make_dev_reward_fn
        ds, eps = grpo_dataset(data_dir, model, limit)
        reward_fn = make_dev_reward_fn(eps, arm)  # unverified DEV training reward only
    cfg = GRPOConfig(output_dir=out, max_steps=steps, num_generations=group, max_completion_length=max_new, learning_rate=lr,
                     per_device_train_batch_size=group, generation_batch_size=group * prompts_per_step,
                     gradient_accumulation_steps=1, beta=0.0, loss_type="dapo", temperature=1.0, bf16=True, logging_steps=1,
                     save_strategy="no", report_to="none", remove_unused_columns=False, gradient_checkpointing=True, seed=seed)
    tm = Timing()
    m = _model(model, sft_adapter, True)
    kw = {} if sft_adapter else dict(peft_config=LoraConfig(**LORA))
    tr = GRPOTrainer(model=m, args=cfg, reward_funcs=reward_fn, train_dataset=ds, processing_class=load_tok(model),
                     callbacks=[tm.callback()], **kw)
    tm.wrap(tr, "_generate_and_score_completions", "rollout_and_score")
    tm.wrap(tr, "training_step", "training_step_total")
    t0 = time.perf_counter()
    tr.train()
    wall = time.perf_counter() - t0
    tr.model.save_pretrained(f"{out}/adapter")
    keys = ("step", "reward", "reward_std", "frac_reward_zero_std", "grad_norm", "loss", "learning_rate", "completions/mean_length",
            "completions/clipped_ratio", "kl")
    return dict(stage="grpo", arm=arm, wall_s=wall, peak_gb=peak_gb(), steps=steps, group=group, max_new=max_new, lr=lr, timing=tm.report(),
                optimizer_steps=tr.state.global_step, adapter_sha256=sha256_dir(f"{out}/adapter"),
                delta_vs_sft=delta_norm(sft_adapter, f"{out}/adapter") if sft_adapter else None,
                log=[{k: x[k] for k in keys if k in x} for x in tr.state.log_history if "reward" in x])


def run_eval(model, data_dir, adapter=None, g=4, max_new=400, limit=0, batch=8, seed=0, variant="none") -> dict:
    import torch
    from modal_train.core import data_prompt_version, eval_row, render_prompt, summarize, variant_episode
    from rl.first_update import strip_special
    eps = [variant_episode(e, variant) for e in load_eps(data_dir, "dev")]  # variant: CONTROL prompt perturbation (eval only)
    eps = eps[:limit] if limit else eps
    raw = (Path(data_dir) / "raw_only").exists()   # sealed TEST: no outcome table exists, store raw completions only (scored later)
    tok = load_tok(model)
    pv = data_prompt_version(data_dir)
    m = _model(model, adapter, False).eval()
    torch.manual_seed(seed)
    rows, ntok, t0 = [], 0, time.perf_counter()
    for i in range(0, len(eps), batch):
        chunk = eps[i:i + batch]
        enc = tok([render_prompt(model, tok, e[1], pv) for e in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=max_new, num_return_sequences=g,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j in range(gen.shape[0]):
            ids = gen[j]; n = int((ids != tok.pad_token_id).sum())
            ntok += n
            txt = tok.decode(ids, skip_special_tokens=False)
            rows.append(dict(kind=chunk[j // g][0].kind, text=strip_special(txt), ntok=n) if raw
                        else eval_row(txt, n, max_new, chunk[j // g], chunk[j // g][0].kind))
    wall = time.perf_counter() - t0
    return dict(stage="eval", split="dev", prompt_version=pv, adapter=adapter, variant=variant, g=g, wall_s=wall, gen_tok_s=ntok / wall, peak_gb=peak_gb(),
                summary=None if raw else summarize(rows, max_new), completions=rows)
