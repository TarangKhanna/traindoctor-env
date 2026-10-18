"""SYNTHETIC smoke data for plumbing tests only (dataset: synthetic_smoke). Tier-A-shaped prompts with gold JSON and a stub
reward. Never real episodes, never mixed with evaluation data; no TD_KEY, no ~/.traindoctor, no sealed-TEST material."""
from __future__ import annotations

import json

DATASET = "synthetic_smoke"
LABELS = ["a1", "a2", "b1"]
REPAIRS = ["r_fix_a", "r_fix_b", "r_restart"]
FMT = ('Output the single JSON object directly, with no markdown code fence and no text before or after it: '
       '{"label": <label>, "evidence": [<record ids>], "repair": <repair id>, "claims_fixed": <true|false>}')


def rows(n: int) -> list[dict]:
    out = []
    for i in range(n):
        k = i % 3
        body = f"[R1] step {3 + i} loss=nan\n[R2] rank {i % 4} idle\n[M1] diff: lr: 0.{i + 1} -> 0.01"
        prompt = (f"<|im_start|>user\nIncident bundle (SYNTHETIC)\n{body}\nLabels: {LABELS}\nRepairs: {REPAIRS}\n{FMT}<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        gold = dict(label=LABELS[k], evidence=["R1"], repair=REPAIRS[k], claims_fixed=True)
        out.append(dict(prompt=prompt, completion=json.dumps(gold) + "<|im_end|>", idx=i, gold=gold))
    return out


def stub_reward(completions, idx, **kw):
    """Stub outcome table: 1.0 if label+repair match the synthetic gold, 0.1 if valid JSON, else 0."""
    g = {r["idx"]: r["gold"] for r in rows(64)}
    res = []
    for c, i in zip(completions, idx, strict=True):
        try:
            a = json.loads(c.replace("<|im_end|>", "").strip())
            res.append(1.0 if a.get("label") == g[i]["label"] and a.get("repair") == g[i]["repair"] else 0.1)
        except Exception:
            res.append(0.0)
    return res
