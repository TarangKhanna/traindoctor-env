# Tests

Counts below are from `PYTHONPATH=. python -m pytest -q -rs -rf -m "..."` run from a clean `git clone` with `requirements-cpu.txt` installed (torch 2.14.1 CPU), macOS arm64, 2026-10-19. `pytest.ini` deselects `sweep` by default and registers the `executed` marker.

| selection | Python | result |
|---|---|---|
| `-m "not sweep and not executed"` (what CI runs), repo venv with `transformers`, 2026-10-19, current branch | 3.12 | **199 passed, 14 skipped, 117 deselected** |
| same selection, clean clone without `transformers`, before the quickstart CLI tests were added (`332782b`) | 3.12.9 and 3.13.2 | 176 passed, 17 skipped, 117 deselected on each; not re-run on the current branch |
| `-m "executed and not sweep"` under the `sandbox-exec` profile, GitHub-hosted `macos-14` runner, Python 3.12 (python.org framework build), workflow `executed-macos` | 3.12 | **Not consistently passing.** All observed runs, 2026-10-19/20: (1) before the framework fix, 51 passed / 64 failed (the interpreter could not start under the profile); (2) branch `executed-macos` at `1d037a7`, 115 passed in 706 s; (3) `main` at `f36a0f6` (run 39297918958), 72 passed / 43 failed in 1508 s, the first scenario (A1_src) passed and later scenarios failed, cause not established; (4) rerun on `main` (run 39302209373) had not finished when this was written |
| same selection on a developer Mac, other Python builds (uv, Homebrew), or Python 3.13 | | not run |
| `-m sweep` | 3.12 | 2 collected; not run |

The two rows are separate observed runs and are not comparable: the extra skips in the clean clone are `transformers` imports in `tests/test_token_cap.py`. Skipped tests are not passing tests. CI runs this selection on Python 3.12 and 3.13.

To run the executed tests, use a host where `sandbox-exec` can be applied (an unrestricted macOS shell), or a configured Modal account: `PYTHONPATH=. python -m pytest -q -rs -rf -m "executed and not sweep"`. The `executed-macos` workflow runs the same selection on a GitHub-hosted macOS runner. Its first run failed 64 of 115 because the profile did not allow a framework Python's second binary (`Resources/Python.app/Contents/MacOS/Python`), so no job could start; with that rule added the run passed. That is one run on one runner and Python build, not evidence about other hosts.

- The 18 executed failures all raise `jobs.sandbox.SandboxUnavailable: job isolation was requested but sandbox-exec cannot be applied here`. These tests launch jobs under the macOS `sandbox-exec` profile, which cannot be nested inside the restricted shell this run used. They are environmental, but they have not been re-run on an unrestricted macOS machine, so the cause is inferred from the error text. The local sandbox is macOS-only; CI does not run the `executed` tests.
- Skipped in the first row: `tests/test_sandbox.py` (needs `sandbox-exec`) and `tests/test_token_cap.py` (needs `sandbox-exec` for the verified outcome tables, or `transformers` and the Qwen3 tokenizer in the offline Hugging Face cache).
- `-m sweep` (bundle size and validity sweep over every variant) was not run.
- The Modal backend is exercised for real only by a live run: `scripts/quickstart.py --modal` (about $0.03, needs a Modal account). `tests/test_modal_sandbox_backend.py` covers the backend without Modal.

Linux: the `executed` tests are not part of the required CI because `sandbox-exec` does not exist there and the sandboxed verification path needs it. The non-required workflow `.github/workflows/executed-linux.yml` (manual dispatch) runs the Gloo subset with `CI=true TD_CI_UNSANDBOXED=1`; it has not been run, and tests that need OS isolation skip there.

First `executed-linux` run (unsandboxed GitHub-hosted Ubuntu runner, 115 tests selected by `-m "executed and not sweep"`): 49 passed, 60 skipped (sandbox-exec), 6 failed, 195 deselected. The 6 failures all assert that a legitimate repair is verified (`ok=True`); `reexec.mark_sandbox` forces `ok=False` for any unsandboxed re-execution, so they cannot pass without OS isolation. They now skip when no sandbox is available. A macOS host run of the full selection has not been done, so the real-launcher suite is not claimed as passing.


The failed runs report `FixOutcome(ok=False)` and downstream assertion failures without the worker stderr, watchdog reason or launcher termination reason, so the cause of run (3) is unknown. The launcher has a startup grace period and a maximum runtime, so timing sensitivity on a loaded runner is plausible but not shown. Native macOS execution is an optional, experimental development path (the README calls the profile a development convenience, not a security boundary); no clean executed-suite claim is made, and the framework-Python profile fix and its tests are kept. If verified local macOS execution becomes a requirement, the next step is to capture the first failing rank's stderr, exit status, launcher termination reason and trusted job record before changing anything.

Current-code end-to-end validation does not depend on this: `scripts/modal_regression.py` replays a fixed set of gold, no-op and incorrect repairs in Modal gVisor sandboxes (see [MODAL_REGRESSION.md](MODAL_REGRESSION.md)).
