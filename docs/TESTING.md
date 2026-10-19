# Tests

Counts below are from `PYTHONPATH=. python -m pytest -q -rs -rf -m "..."` run from a clean `git clone` with `requirements-cpu.txt` installed (torch 2.14.1 CPU), macOS arm64, 2026-10-19. `pytest.ini` deselects `sweep` by default and registers the `executed` marker.

| selection | Python | result |
|---|---|---|
| `-m "not sweep and not executed"` (what CI runs), repo venv with `transformers`, 2026-10-19, current branch | 3.12 | **193 passed, 14 skipped, 117 deselected** |
| same selection, clean clone without `transformers`, before the quickstart CLI tests were added (`332782b`) | 3.12.9 and 3.13.2 | 176 passed, 17 skipped, 117 deselected on each; not re-run on the current branch |
| `-m "executed and not sweep"` (real multi-process Gloo jobs) | 3.12 | 115 collected; not run in the restricted shell used for this check |
| `-m sweep` | 3.12 | 2 collected; not run |

The two rows are separate observed runs and are not comparable: the extra skips in the clean clone are `transformers` imports in `tests/test_token_cap.py`. Skipped tests are not passing tests. CI runs this selection on Python 3.12 and 3.13.

To run the executed tests, use a host where `sandbox-exec` can be applied (an unrestricted macOS shell), or a configured Modal account: `PYTHONPATH=. python -m pytest -q -rs -rf -m "executed and not sweep"`.

- The 18 executed failures all raise `jobs.sandbox.SandboxUnavailable: job isolation was requested but sandbox-exec cannot be applied here`. These tests launch jobs under the macOS `sandbox-exec` profile, which cannot be nested inside the restricted shell this run used. They are environmental, but they have not been re-run on an unrestricted macOS machine, so the cause is inferred from the error text. The local sandbox is macOS-only; CI does not run the `executed` tests.
- Skipped in the first row: `tests/test_sandbox.py` (needs `sandbox-exec`) and `tests/test_token_cap.py` (needs `sandbox-exec` for the verified outcome tables, or `transformers` and the Qwen3 tokenizer in the offline Hugging Face cache).
- `-m sweep` (bundle size and validity sweep over every variant) was not run.
- The Modal backend is exercised for real only by a live run: `scripts/quickstart.py --modal` (about $0.03, needs a Modal account). `tests/test_modal_sandbox_backend.py` covers the backend without Modal.

Linux: the `executed` tests are not part of the required CI because `sandbox-exec` does not exist there and the sandboxed verification path needs it. The non-required workflow `.github/workflows/executed-linux.yml` (manual dispatch) runs the Gloo subset with `CI=true TD_CI_UNSANDBOXED=1`; it has not been run, and tests that need OS isolation skip there.

First `executed-linux` run (unsandboxed GitHub-hosted Ubuntu runner, 115 tests selected by `-m "executed and not sweep"`): 49 passed, 60 skipped (sandbox-exec), 6 failed, 195 deselected. The 6 failures all assert that a legitimate repair is verified (`ok=True`); `reexec.mark_sandbox` forces `ok=False` for any unsandboxed re-execution, so they cannot pass without OS isolation. They now skip when no sandbox is available. A macOS host run of the full selection has not been done, so the real-launcher suite is not claimed as passing.
