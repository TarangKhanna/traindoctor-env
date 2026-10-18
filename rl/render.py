"""Per-model prompt rendering for screens. qwen3 = our hybrid template with thinking off (tiera.training_prompt);
chat = the model's OWN tokenizer chat template over the same messages (Qwen3-4B-Instruct-2507, Gemma). No golden test
for the 'chat' path: it is the tokenizer's own output, not bytes we own."""
from __future__ import annotations

from traindoctor_env.tiera import messages, training_prompt

KIND = {"mlx-community/Qwen3-0.6B-4bit": "qwen3", "mlx-community/Qwen3-4B-4bit": "qwen3"}


def render(model_id: str, tok, bundle, version: str = "v1") -> str:
    """version: tiera prompt version ("v1" = original, kept for reproducing old results; "v2" = neutral framing + NO_FAULT
    definition). v1 calls are made exactly as before (no version argument)."""
    kw = {} if version == "v1" else {"version": version}
    if KIND.get(model_id, "chat") == "qwen3":
        return training_prompt(bundle, **kw)
    return tok.apply_chat_template(messages(bundle, **kw), tokenize=False, add_generation_prompt=True)
