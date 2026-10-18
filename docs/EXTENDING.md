# Using the packages and adding a task

Install (CPU, Python 3.12): see [Quickstart](QUICKSTART.md). `pip install ".[dev]"` or `pip install git+https://github.com/TarangKhanna/traindoctor-env` makes the packages importable.

Entry points:
- `traindoctor_env/`: scenario generation, snapshot/replay, tools, Tier A bundle, verifiers V0-V2, oracle. Plain Python; no OpenEnv, Inspect or MLX imports.
- `jobs/`: the executable Gloo job, fault templates, launcher, re-execution harness.
- `traindoctor_env/modal_backend.py`: runs a replay in a Modal Sandbox (`pip install ".[modal]"`).
- `scripts/quickstart.py`: build a failed job, show the agent view, apply the gold repair, re-run.
- `scripts/redteam_suite.py`: run fake and gold repairs through the verifier.

Add a task: write a fault template in `jobs/` that injects the failure into the training job, add its kind to the scenario generator and an evidence predicate and gold repair to the oracle, then add red-team cases (a repair the verifier should reject) before using it as a reward.

Training loops:
- **Tier A** is a single-turn bundle with a repair menu: `traindoctor_env/bundle.py` builds the prompt text, `tiera.py` renders the chat prompt, parses the answer and scores it (V0/V1/V2). Training code is not part of this repository; the SFT and GRPO scripts used for the reported results are in [traindoctor-research](https://github.com/TarangKhanna/traindoctor-research).
- **Tier B** is multi-turn: the agent edits the job's declared editable files and the diff is replay-verified.
- Adapters for external RL frameworks are not published yet.
