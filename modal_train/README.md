# modal_train: CUDA port of SFT / GRPO / eval

Core packages never import Modal. The macOS MLX path (`rl/*.py`) is separate. This directory imports `rl.render`, `rl.first_update` (classify, strip_special) and `traindoctor_env.tiera.reward` (strict gate, V0/V2); prompts are produced only through `rl.render` (HF Qwen3 ids alias to the MLX ids so the same thinking-off template is used).

Pins: modal client 1.3.5, torch 2.14.1, transformers 5.20.1, trl 1.16.0, peft 0.21.2, accelerate 1.15.0, Python 3.12 `debian_slim` image. GPU strings `A100-80GB` (default) / `H100` (`TD_GPU=H100`).

## Files
`app.py` (image, volumes, functions, entrypoints) · `trainlib.py` (TRL/PEFT code) · `core.py` (pure glue) · `prepare.py` (local episodes) · `cpu_app.py` (CPU scoring and sandbox replay) · `sandbox_backend.py` (Modal Sandbox replay backend) · `budget.py` + `rates.json` (cost guard) · `run.sh`, `run_cpu.sh` (launchers).

## Flow
1. Local, CPU, no Modal: `python -m modal_train.prepare --dev 48 --train 64` (real re-execution; DEV and TRAIN seeds only, asserted disjoint from the sealed TEST seeds). Writes `modal_train/cache/` (gitignored).
2. `modal_train/run.sh 300 upload`, then `run_smoke`, `run_sft "--run-id sft1"`, `run_grpo "--run-id g_v2 --arm V2 --sft-run sft1"`, `run_eval "--run-id e1 --adapter-run g_v2"`, then `modal run modal_train/app.py::download`.

Evaluation uses HF `generate` (sampling T=1.0, top_p 0.95), not the MLX sampler, so numbers are not directly comparable to MLX runs. `run.sh` forwards extra arguments as one string; quote as shown. The training and evaluation runs behind the published tables were executed on Modal; the run scripts are included, the per-run logs are not.
