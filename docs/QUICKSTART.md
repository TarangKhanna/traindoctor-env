# Quickstart (full)

Commands are copy-pasteable from a fresh clone.

CPU only, no paid API, no Modal needed for steps 1-2. Python 3.12.

```bash
git clone https://github.com/TarangKhanna/traindoctor-env && cd traindoctor-env
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt      # torch 2.14.1, numpy, pytest, hypothesis (pins equal uv.lock)
#   or, to import the packages elsewhere:  pip install ".[dev]"   /   pip install git+https://github.com/TarangKhanna/traindoctor-env
#   extras: ".[modal]" (Modal backend). The scenario generator needs TD_KEY (any random hex) in the environment; quickstart.py sets one for you.
#   Linux: pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt

# 1-3: build the failed 4-rank job, show what the agent sees, apply the gold repair, re-run locally (about 30 s)
python scripts/quickstart.py --fake F11 --out quickstart_out
```

This runs a real 4-process Gloo job, prints the agent-visible records, the gold diff, and the local re-run with all invariants. The local re-run is labelled **local, UNVERIFIED (not sandboxed)**: the verifier forces `ok=False` and prints `INVALID: unsandboxed re-execution`, even though every invariant passes. That refusal is the intended behaviour, not a bug. 

**Verified path (optional, needs a Modal account, about $0.03):** replay the same scenario (seed 123, task A1) with the same two repairs the local run showed, the gold repair and the fake F11 repair, in a Modal Sandbox.

```bash
pip install modal && modal token new
python scripts/quickstart.py --modal         # same --seed and --fake as the local demo; or directly:
TD_KEY=$(python -c "import secrets;print(secrets.token_hex(16))") \
  python scripts/redteam_suite.py --run quickstart --seed 123 --cases gold,F11 --cap-usd 0.15
```

Expected: `gold accepted=True` with `exec_mode=modal_sandbox`, `F11 accepted=False` (rejected by the recovery invariants). `--seed` and `--fake` apply to both the local and the Modal run, so the pair stays identical. This seed-123 replay has not been recorded. The recorded sandboxed results are for other instances: the A1_src and A2 rows of the research repository's `evidence/results/REDTEAM_TABLE.md` and `quickstart_verified_replay.txt`. A normally installed Modal client is found automatically; `TD_MODAL_SITE` (a `site-packages` directory) is only an optional override for a client installed in a separate environment. The sandbox is network-less; the host process needs only the Modal API endpoints and `*.modal-storage.com` for downloads.

Unit tests:

```bash
python -m pytest -q -m "not sweep and not executed"
python -m pytest -q -m "executed and not sweep"   # real Gloo jobs, slower
```

See [Known test failures](TESTING.md) for the exact current counts.
