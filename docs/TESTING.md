# Tests

Three levels, which are not interchangeable:

1. **Unit tests** (CI, every push): `-m "not sweep and not executed"`. No real multi-process job, no sandbox, no Modal. A green CI badge means only this level passed.
2. **Executed integration tests** (`-m executed`, 115 tests): real multi-process Gloo jobs through the launcher. Not in required CI and not consistently passing on any host (below).
3. **Sandboxed verification** (`scripts/modal_regression.py`): repairs replayed in network-less Modal gVisor sandboxes against the verifier. Paid, run manually; the last recorded run is below.

Counts come from `PYTHONPATH=. python -m pytest -q -rs -rf -m "..."` with `requirements-cpu.txt` (torch 2.14.1 CPU) on macOS arm64 unless noted. `pytest.ini` deselects `sweep` by default and registers `executed`. Skipped tests are not passes.

| selection | Python | historical result |
| --- | --- | --- |
| `-m "not sweep and not executed"` (CI selection), repo venv with `transformers`, branch `release-ready` (5ab56fc + relaunch-detail change), 2026-10-21 | 3.12.9 | **514 passed, 14 skipped, 117 deselected** (115 executed + 2 sweep). Skips: 8 `tests/test_sandbox.py` and 6 `tests/test_token_cap.py`, all because `sandbox-exec` cannot be applied in the restricted shell used |
| Same selection, clean clone without `transformers`, before quickstart CLI tests (`332782b`) | 3.12.9 and 3.13.2 | 176 passed, 17 skipped, 117 deselected on each; not re-run on the later branch |
| `-m "executed and not sweep"`, `sandbox-exec` profile, GitHub-hosted `macos-14`, python.org framework build, workflow `executed-macos` | 3.12 | **Not consistently passing.** Observed 2026-10-19/20: (1) before the framework fix, 51 passed / 64 failed (interpreter could not start); (2) branch `executed-macos`, `1d037a7`, 115 passed in 706 s; (3) `main`, `f36a0f6`, run 39297918958, 72 passed / 43 failed in 1508 s: first scenario A1_src passed, later scenarios failed, cause not established; (4) `main` rerun 39302209373: 68 passed / 47 failed in 1996 s. Runs (2)-(4) differ in code only by a docstring and a guard that cannot apply on the runner (`git diff 1d037a7 f36a0f6 -- jobs traindoctor_env`), so the failures depend on the runner or run, not on a code change; the failing mechanism is not established |
| Same executed selection on a developer Mac, other Python builds (uv, Homebrew), or Python 3.13 | | Not run |
| `-m sweep` (bundle size and validity over every variant) | 3.12 | 2 collected; not run |

The first two rows are separate, non-comparable runs; the clean clone's extra skips are `transformers` imports in `tests/test_token_cap.py`. The first row skips `tests/test_sandbox.py` (needs `sandbox-exec`) and `tests/test_token_cap.py` (needs `sandbox-exec` for verified outcome tables, or `transformers` and the Qwen3 tokenizer in the offline Hugging Face cache).

## Executed tests and host limitations

Run `PYTHONPATH=. python -m pytest -q -rs -rf -m "executed and not sweep"` from an unrestricted macOS shell that permits `sandbox-exec`, or with a configured Modal account. The `executed-macos` workflow uses the same selection. Its initial 64 of 115 failures came from the profile blocking a framework Python's second binary (`Resources/Python.app/Contents/MacOS/Python`); adding that rule produced run (2). One runner and Python build do not establish support for other hosts.

An earlier restricted-shell run reported 18 executed failures, all `jobs.sandbox.SandboxUnavailable: job isolation was requested but sandbox-exec cannot be applied here`. Nested sandbox restrictions are inferred from the error text; these failures were not re-run on an unrestricted Mac. The native sandbox is macOS-only, and required CI excludes executed tests.

Linux lacks `sandbox-exec`. The optional, manually dispatched `.github/workflows/executed-linux.yml` runs the Gloo subset with `CI=true TD_CI_UNSANDBOXED=1`; tests requiring OS isolation skip. Initially unrun, its first unsandboxed GitHub-hosted Ubuntu run selected 115 tests: 49 passed, 60 skipped (`sandbox-exec`), 6 failed, 195 deselected. The 6 failures required a legitimate repair to return `ok=True`, while `reexec.mark_sandbox` forces `ok=False` for unsandboxed re-execution. Those tests now skip without a sandbox. At that stage, no macOS host run of the full selection had been done; the real-launcher suite was not claimed as passing.

Failed runs report `FixOutcome(ok=False)` and downstream assertion failures without worker stderr, watchdog reason or launcher termination reason. The cause of run (3) remains unknown. Launcher startup grace and maximum runtime make timing sensitivity on a loaded runner plausible, but unproven. Native macOS execution remains an optional, experimental development path: its profile is a development convenience, not a security boundary. The framework-Python fix and tests remain; there is no clean executed-suite claim. Further diagnosis requires the first failing rank's stderr, exit status, launcher termination reason and trusted job record before changes.

`tests/test_modal_sandbox_backend.py` tests the backend without Modal. A live `scripts/quickstart.py --modal` run exercises it for real (about $0.03, requires a Modal account). The fixed regression below provides separate current-code end-to-end validation.

## Modal regression suite

`scripts/modal_regression.py` replays current code against the frozen verifier in network-less Modal gVisor sandboxes: A1_src seed 90005, A1_cfg 90014, A2 90003, B_SKIP 90001, B_DUP 90000. Each kind must accept gold and reject null plus two repairs from `scripts/redteam_suite.py`: A kinds use F11 (`world_size=1`) and F03 (identity `all_reduce`); B kinds use F16 (restart from scratch) and F09 (zeroed checkpoint model). Exit 0 requires every expectation and `exec_mode=modal_sandbox` on every row; verifier, invariants and timeouts are unchanged.

About 20 replays, about $0.07, requires Modal authentication:

```bash
SSL_CERT_FILE=/etc/ssl/cert.pem TD_KEY=00ff PYTHONPATH=. python scripts/modal_regression.py --cap-usd 0.5
```

Writes `regression_results/modal_regression.{md,json}`; `--list` prints cases without execution. **Recorded result:** code `f36a0f61bd005823f71e94c02bae4d4fac52f92d`, 2026-10-19 (UTC), Python 3.12.9, Modal client 1.6.2, script spend $0.074: 20 of 20 expectations held, 0 violations; 5 gold accepted, 5 null and 10 incorrect repairs rejected, all `modal_sandbox`. First rejecting invariants: A null/F03 = I1 `no_completed_relaunch`; A F11 = I6 `world_size`; B null/F16 = sample gaps and dups; B F09 = I4 `ranks_inconsistent`. [Full table](../regression_results/modal_regression.md). In the JSON, the local virtualenv prefix inside recorded rank tracebacks is redacted to `<VENV>` (2026-10-21); nothing else was changed.

**Coverage limits:** not the 115 executed tests; one seed and 4 cases per kind, not the full cheat catalog; one run, with no repeat-run flakiness estimate. Spend is the script's estimate, not a Modal bill. The optional native macOS sandbox is not exercised.
