# Using the packages and adding a task

Moved from the top-level README.

Install (CPU, Python 3.12): see [Quickstart](QUICKSTART.md). `pip install ".[dev]"` or `pip install git+https://github.com/TarangKhanna/traindoctor-env` makes the packages importable.

Entry points:
- `traindoctor_env/`: scenario generation, snapshot/replay, tools, Tier A bundle, verifiers V0-V2, oracle. Plain Python; no OpenEnv, Inspect or MLX imports.
- `jobs/`: the executable Gloo job, fault templates, launcher, re-execution harness.
- `scripts/quickstart.py`: build a failed job, show the agent view, apply the gold repair, re-run.
- `scripts/redteam_suite.py`: run fake and gold repairs through the verifier.

Add a task: write a fault template in `jobs/` that injects the failure into the training job, add its kind to the scenario generator and an evidence predicate and gold repair to the oracle, then add red-team cases (a repair the verifier should reject) before using it as a reward.

Training loops:
- **Tier A** is a single-turn bundle with a repair menu. Prompts and splits come from `rl/`; SFT and GRPO helpers and the Modal backend are in `rl/` and `modal_train/`.
- **Tier B** is multi-turn: the agent edits the job's declared editable files and the diff is replay-verified.
- Adapters for external RL frameworks are not published yet.
