# modal_train: CUDA port of SFT / GRPO / eval (isolated from the core env)

Core packages never import Modal. The macOS MLX path (`rl/*.py`) is untouched. This directory imports `rl.render`,
`rl.first_update` (classify, strip_special) and `traindoctor_env.tiera.reward` (strict gate, V0/V2); prompts are produced
only through `rl.render` at runtime (HF Qwen3 ids alias to the MLX ids so the same thinking-off template is used).
Oracle SFT targets use the same recipe as `rl/gen_sft.py` (reimplemented inside `prepare.py` only to also keep the
fix-outcome tables in one pass; the target construction lines are identical).

## Pins (chosen 2026-10-18 from PyPI / docs)
modal client 1.3.5 (installed locally), torch 2.14.1 (matches pyproject), transformers 5.20.1, trl 1.16.0, peft 0.21.2,
accelerate 1.15.0, python 3.12 image `debian_slim`. GPU strings `A100-80GB` (default) / `H100` (`TD_GPU=H100`; Modal may
upgrade to H200 unless `H100!`).

## Files
`app.py` (image, volumes `td-hf-cache` and `td-results`, functions with explicit `timeout=`, entrypoints) ·
`trainlib.py` (TRL/PEFT code) · `core.py` (pure glue) · `budget.py` + `rates.json` (spend guard) ·
`prepare.py` (local episodes) · `run.sh` (backgrounded launcher) · `tests/test_modal_train.py`.

## Flow
1. Local, CPU, no Modal: `uv run python -m modal_train.prepare --dev 48 --train 64` (real re-execution; DEV and TRAIN seeds only,
   asserts disjoint from sealed TEST). Writes `modal_train/cache/` (gitignored).
2. `modal_train/run.sh 300 upload` (volume upload, no compute).
3. `modal_train/run.sh 2000 run_smoke` (0.6B: 3 SFT steps + 1 GRPO step; reports VRAM, tok/s, rollout/backward/optimizer seconds).
4. `run.sh 4000 run_sft "--run-id sft1"`, `run.sh 15000 run_grpo "--run-id g_v2 --arm V2 --sft-run sft1"`,
   `run.sh 4000 run_eval "--run-id e1 --adapter-run g_v2"`, then `modal run modal_train/app.py::download`.
Budget: each launch writes a projected row (cost of the full function timeout) to `modal_train/ledger.jsonl`, refuses above
$30 total, and replaces it with the actual container seconds on return. Cap/rates are config (`budget.CAP_USD`, `rates.json`).

## Unverified (nothing has been run on Modal or a GPU)
- `rates.json` ($2.50/h A100-80GB, $3.95/h H100) is from the brief, not `modal billing rates`; spend limits unchecked.
- Modal auth not set up; image build, volume paths, `batch_upload`/`read_file` behavior untested. `modal run --help` parses (1.3.5).
- TRL 1.16 behavior untested: SFT prompt/completion + `completion_only_loss`, GRPO accepting a PeftModel loaded with
  `is_trainable=True`, `loss_type="dapo"`, string-prompt tokenization parity with the MLX path, 4B + group 8 fits in 80 GB.
- Per-phase timings hook private TRL attrs (`_generate_and_score_completions`, `training_step`); `report()["hooked"]` lists what attached.
- Eval uses HF `generate` (sampling T=1.0, top_p 0.95), not the MLX sampler, so numbers are not directly comparable to MLX runs.
- `run.sh` forwards extra args unquoted via one string; quote as shown.
- The prompt/menu are changing in another branch: re-run `prepare` after that merges (cached bundles embed the old menu).
