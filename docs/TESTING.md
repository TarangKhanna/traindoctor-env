# Tests

Run from a fresh venv built from `requirements-cpu.txt` (macOS arm64, Python 3.12, torch 2.14.1 CPU), 2026-10-19, after trimming the repository to the library:

| selection | result |
|---|---|
| `pytest -m "not sweep and not executed"` (what CI runs) | **179 passed, 14 skipped** |
| `pytest -m "executed and not sweep"` (real multi-process Gloo jobs) | **37 passed, 18 failed, 60 skipped** |

- The 18 executed failures all raise `jobs.sandbox.SandboxUnavailable: job isolation was requested but sandbox-exec cannot be applied here`. These tests launch jobs under the macOS `sandbox-exec` profile, which cannot be nested inside the restricted shell this run used. They are environmental, but they have not been re-run on an unrestricted macOS machine, so the cause is inferred from the error text. The local sandbox is macOS-only; CI does not run the `executed` tests.
- Skipped in the first row: `tests/test_sandbox.py` (needs `sandbox-exec`) and `tests/test_token_cap.py` (needs `sandbox-exec` for the verified outcome tables, or `transformers` and the Qwen3 tokenizer in the offline Hugging Face cache).
- `-m sweep` (bundle size and validity sweep over every variant) was not run.
- The Modal backend is exercised for real only by a live run: `scripts/quickstart.py --modal` (about $0.03, needs a Modal account). `tests/test_modal_sandbox_backend.py` covers the backend without Modal.
