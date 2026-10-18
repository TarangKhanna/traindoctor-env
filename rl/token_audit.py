"""True prompt token counts (model's own tokenizer) vs bundle.tokens (4-chars/token estimate, CAP_TOKENS=3072 (2048 when audited))."""
import json, pickle, statistics as st, sys
from transformers import AutoTokenizer
from rl.render import render
sys.path.insert(0, ".")
eps = pickle.loads(open("out/pilot16/episodes.pkl", "rb").read())
train = [json.loads(l) for l in open("out/sft/train.jsonl")] if len(sys.argv) > 1 else []
out = {}
for m in ["mlx-community/Qwen3-0.6B-4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit", "mlx-community/gemma-4-e2b-it-4bit"]:
    tok = AutoTokenizer.from_pretrained(m)
    n = [len(tok.encode(render(m, tok, e[1]))) for e in eps]
    bt = [len(tok.encode(e[1].text)) for e in eps]
    out[m] = dict(dev16_prompt_tokens=n, dev16_bundle_text_tokens=bt, est=[e[1].tokens for e in eps],
                  over_2048_bundle=sum(x > 2048 for x in bt), min=min(bt), median=st.median(bt), max=max(bt))
    if train and "0.6B" in m:
        t = [len(tok.encode(r["prompt"])) for r in train]
        out["train_qwen3_prompt_tokens"] = dict(n=len(t), min=min(t), median=st.median(t), max=max(t), over_2048=sum(x > 2048 for x in t))
    print(m, {k: v for k, v in out[m].items() if k in ("over_2048_bundle", "min", "median", "max")})
json.dump(out, open("evidence/archive/dev_json/token_audit.json", "w"), indent=1)
