"""LOCAL, no re-execution: derive modal_train/cache_v2prompt from modal_train/cache. Same episodes, outcome tables and
oracle targets (the fix table does not depend on the prompt); only a `prompt_version` marker differs, which makes
trainlib render the v2 prompt. Checks every bundle still fits CAP_TOKENS (estimator, and the real Qwen3 tokenizer if
`--tok` is given)."""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

from rl.render import render
from traindoctor_env.bundle import CAP_TOKENS, est_tokens, PROMPT_OVERHEAD
from traindoctor_env.tiera import prompt_text

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "cache"))
    p.add_argument("--dst", default=str(ROOT / "cache_v2prompt"))
    p.add_argument("--tok", default="", help="HF tokenizer id/path for the real-token check (optional)")
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(exist_ok=True)
    tok = None
    if a.tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tok, local_files_only=True)
    for f in sorted(src.glob("*")):
        if f.name != "PROVENANCE.md":
            shutil.copyfile(f, dst / f.name)
    (dst / "prompt_version").write_text("v2\n")
    (dst / "PROVENANCE.md").write_text(f"Derived from {src} (same eps/tables/oracle targets); prompt_version=v2 marker only. "
                                       "Prompt v2 = neutral framing + NO_FAULT definition (traindoctor_env.tiera.reframe_v2).\n")
    worst = {}
    for name in ("train", "dev"):
        eps = pickle.loads((src / f"{name}_eps.pkl").read_bytes())      # our own file, written by modal_train.prepare
        for e in eps:
            b = e[1]
            v1, v2 = render("mlx-community/Qwen3-4B-4bit", None, b), render("mlx-community/Qwen3-4B-4bit", None, b, "v2")
            row = dict(est=est_tokens(prompt_text(b, "v2")) + PROMPT_OVERHEAD, real=len(tok.encode(v2)) if tok else None,
                       real_v1=len(tok.encode(v1)) if tok else None)
            for k, v in row.items():
                if v is not None and v > worst.get((name, k), (-1,))[0]:
                    worst[(name, k)] = (v, e[0].kind)
    for k, (v, kind) in sorted(worst.items()):
        print(k, v, kind)
    for (name, k), (v, kind) in worst.items():
        # CAP_TOKENS is defined in real Qwen3 tokens; the estimator is only an upper bound (its v2 value may exceed the cap by the ~34-token header growth while real tokens fit)
        if k == ("real" if tok else "est"):
            assert v <= CAP_TOKENS, f"{name} {k} {v} > {CAP_TOKENS} ({kind})"
    print(f"all <= {CAP_TOKENS}; wrote {dst}")


if __name__ == "__main__":
    main()
