# Tests

**2026-10-31 reproduction:** fresh local clones reproduced the worked recovery
demo and its NumPy recount. A focused selection passed 23 checks but failed
`test_live_job_and_rendezvous_listeners_are_loopback`: the parent listener was
loopback, but the test did not observe the worker listener within its polling
window. The cause remains unresolved; the later successful demo is not proof
that this test or historical Gloo flakiness is fixed. The companion
[reproduction record](https://github.com/TarangKhanna/traindoctor-research/blob/main/docs/REPRODUCIBILITY.md)
separates fresh execution, archived recounts and resource-interrupted Harbor runs.

Three levels, which are not interchangeable:

1. **Unit tests** (CI, every push): `-m "not sweep and not executed"`. No real multi-process job, no sandbox, no Modal. A green CI badge means only this level passed.
2. **Executed integration tests** (`-m executed`; historical run counts below): real multi-process Gloo jobs through the launcher. Not in required CI; recorded native macOS runs are inconsistent and other host coverage is limited, as documented below. Use `pytest --collect-only -q -m "executed and not sweep"` for the current collection; moving integration tests out of the unit selection changed it.
3. **Sandboxed verification** (`scripts/modal_regression.py`): repairs replayed in network-less Modal gVisor sandboxes against the verifier. Paid, run manually; the last recorded run is below.

Counts come from `PYTHONPATH=. python -m pytest -q -rs -rf -m "..."` with `requirements-cpu.txt` (torch 2.14.1 CPU) on macOS arm64 unless noted. `pytest.ini` deselects `sweep` by default and registers `executed`. Skipped tests are not passes.

**2026-10-30 selection correction:** the end-to-end refusal/world-shrink tests and
the snapshot-dependent token-cap test are now marked `executed`. They create real
Gloo scenarios and do not belong in the fast unit selection. Earlier historical
counts below predate this marker correction. Tests remain available under
`-m "executed and not sweep"`; moving them does not turn a skip into a pass or
change their assertions. The unit selection was rerun without loopback privileges.

| selection | Python | historical result |
| --- | --- | --- |
| `-m "not sweep and not executed"` (CI selection), repo venv with `transformers`, branch `release-ready` (5ab56fc + relaunch-detail change), 2026-10-21 | 3.12.9 | **516 passed, 14 skipped, 117 deselected** (115 executed + 2 sweep). Skips: 8 `tests/test_sandbox.py` and 6 `tests/test_token_cap.py`, all because `sandbox-exec` cannot be applied in the restricted shell used |
| Same selection, clean clone without `transformers`, before quickstart CLI tests (`332782b`) | 3.12.9 and 3.13.2 | 176 passed, 17 skipped, 117 deselected on each; not re-run on the later branch |
| `-m "executed and not sweep"`, `sandbox-exec` profile, GitHub-hosted `macos-14`, python.org framework build, workflow `executed-macos` | 3.12 | **Not consistently passing.** Observed 2026-10-19/20: (1) before the framework fix, 51 passed / 64 failed (interpreter could not start); (2) branch `executed-macos`, `1d037a7`, 115 passed in 706 s; (3) `main`, `f36a0f6`, run 39297918958, 72 passed / 43 failed in 1508 s: first scenario A1_src passed, later scenarios failed, cause not established; (4) `main` rerun 39302209373: 68 passed / 47 failed in 1996 s. Runs (2)-(4) differ in code only by a docstring and a guard that cannot apply on the runner (`git diff 1d037a7 f36a0f6 -- jobs traindoctor_env`), so the failures depend on the runner or run, not on a code change; the failing mechanism is not established |
| Same executed selection on a developer Mac, other Python builds (uv, Homebrew), or Python 3.13 | | Not run |
| `-m sweep` (bundle size and validity over every variant) | 3.12 | 2 collected; not run |

The first two rows are separate, non-comparable runs; the clean clone's extra skips are `transformers` imports in `tests/test_token_cap.py`. The first row skips `tests/test_sandbox.py` (needs `sandbox-exec`) and `tests/test_token_cap.py` (needs `sandbox-exec` for verified outcome tables, or `transformers` and the Qwen3 tokenizer in the offline Hugging Face cache).

## Executed tests and host limitations

Run `PYTHONPATH=. python -m pytest -q -rs -rf -m "executed and not sweep"` from an unrestricted macOS shell that permits `sandbox-exec`, or with a configured Modal account. The `executed-macos` workflow uses the same selection. Its initial 64 of 115 failures came from the profile blocking a framework Python's second binary (`Resources/Python.app/Contents/MacOS/Python`); adding that rule produced run (2). One runner and Python build do not establish support for other hosts.

An earlier restricted-shell run reported 18 executed failures, all `jobs.sandbox.SandboxUnavailable: job isolation was requested but sandbox-exec cannot be applied here`. Nested sandbox restrictions are inferred from the error text; these failures were not re-run on an unrestricted Mac. The native sandbox is macOS-only, and required CI excludes executed tests.

Linux lacks `sandbox-exec`. The optional, manually dispatched `.github/workflows/executed-linux.yml` runs the Gloo subset with `CI=true TD_CI_UNSANDBOXED=1`; tests requiring OS isolation skip. Initially unrun, its first unsandboxed GitHub-hosted Ubuntu run selected 115 tests: 49 passed, 60 skipped (`sandbox-exec`), 6 failed, 195 deselected. The 6 failures required a legitimate repair to return `ok=True`, while `reexec.mark_sandbox` forces `ok=False` for unsandboxed re-execution. Those tests now skip without a sandbox. At that stage, no macOS host run of the full selection had been done; the real-launcher suite was not claimed as passing.

**Cause of the macOS failures (2026-10-21, branch `exec-diag`, workflow `exec-diag`).** Same runner type, same code: `release-ready` run 39626012191 gave 64 passed / 51 failed. Replaying each scenario's oracle repair directly passed 6/6 sequentially and in all 4 visible parallel lines (run 39631746321; GitHub shows 10 annotations per step), so the verifier and profile give correct verdicts when the jobs start. Repeated replays failed 16/24 (run 39637475477) and 1/36 (run 39641137684); every failing replay's relaunch has a rank whose Gloo libuv listener got `uv_bind: operation not permitted` (15 of 16 visible in the kept stderr; one truncated), and the kernel logged `deny network-bind local:*:0` at the failure. Gloo binds the first address it finds on `GLOO_SOCKET_IFNAME` (`lo0`: `127.0.0.1`, `::1`, `fe80::1`); under the job profile, binding `127.0.0.1` and `::1` succeeds. Why the chosen address sometimes falls outside the profile's `localhost` rule is not established. Fixing it means widening the profile's bind rule or changing address selection for the ranks, a sandbox decision not taken here. Until then the native macOS executed suite is flaky by this mechanism, not by a verifier defect.

Before the `detail` change, failed runs reported `FixOutcome(ok=False)` and downstream assertion failures without worker stderr, watchdog reason or launcher termination reason. The cause of run (3) remains unknown. Launcher startup grace and maximum runtime make timing sensitivity on a loaded runner plausible, but unproven. Native macOS execution remains an optional, experimental development path: its profile is a development convenience, not a security boundary. The framework-Python fix and tests remain; there is no clean executed-suite claim. Further diagnosis requires the first failing rank's stderr, exit status, launcher termination reason and trusted job record before changes.

`tests/test_modal_sandbox_backend.py` tests the backend without Modal. A live `scripts/quickstart.py --modal` run exercises it for real (about $0.03, requires a Modal account). The fixed regression below provides separate current-code end-to-end validation.

## Modal regression suite

`scripts/modal_regression.py` replays current code against the frozen verifier in network-less Modal gVisor sandboxes: A1_src seed 90005, A1_cfg 90014, A2 90003, B_SKIP 90001, B_DUP 90000. Each kind must accept gold and reject null plus two repairs from `scripts/redteam_suite.py`: A kinds use F11 (`world_size=1`) and F03 (identity `all_reduce`); B kinds use F16 (restart from scratch) and F09 (zeroed checkpoint model). Exit 0 requires every expectation and `exec_mode=modal_sandbox` on every row; verifier, invariants and timeouts are unchanged.

About 20 replays, about $0.07, requires Modal authentication:

```bash
SSL_CERT_FILE=/etc/ssl/cert.pem TD_KEY=00ff PYTHONPATH=. python scripts/modal_regression.py --cap-usd 0.5
```

Writes `regression_results/modal_regression.{md,json}`; `--list` prints cases without execution. **Recorded result:** code `f36a0f61bd005823f71e94c02bae4d4fac52f92d`, 2026-10-19 (UTC), Python 3.12.9, Modal client 1.6.2, script spend $0.074: 20 of 20 expectations held, 0 violations; 5 gold accepted, 5 null and 10 incorrect repairs rejected, all `modal_sandbox`. First rejecting invariants: A null/F03 = I1 `no_completed_relaunch`; A F11 = I6 `world_size`; B null/F16 = sample gaps and dups; B F09 = I4 `ranks_inconsistent`. [Full table](../regression_results/modal_regression.md). In the JSON, the local virtualenv prefix inside recorded rank tracebacks is redacted to `<VENV>` (2026-10-21); nothing else was changed.

**Latest recorded result:** code `6e3685def311ec3f13a406994a18615751713dc4`, 2026-10-21 (UTC), Python 3.12.9, Modal client 1.6.2, script spend $0.100 (Modal billing metered +$0.06, billed $0): 21 of 21 expectations held, 0 violations; the 20 cases above plus a B_SKIP replay with `world_size: 100000`, refused before any worker started, `detail='BAD_CONFIG: world_size must be an integer in [1, 16]'`, `exec_mode=modal_sandbox`. [Table](../regression_results/run-6e3685def311/modal_regression.md); local venv prefix in tracebacks redacted to `<VENV>`.

**Coverage limits:** not the 115 executed tests; one seed and 4 cases per kind, not the full cheat catalog; one run, with no repeat-run flakiness estimate. Spend is the script's estimate, not a Modal bill. The optional native macOS sandbox is not exercised.
