# Modal regression suite

`scripts/modal_regression.py` replays the CURRENT code against the frozen verifier in Modal Sandboxes (gVisor, no network)
for five kinds, one seed each (A1_src 90005, A1_cfg 90014, A2 90003, B_SKIP 90001, B_DUP 90000). Per kind: the gold repair
must be accepted; the null diff and two incorrect repairs from `scripts/redteam_suite.py` must be rejected (A kinds: F11
world_size=1 and F03 identity all_reduce; B kinds: F16 restart-from-scratch and F09 zeroed model in the checkpoint).
Exit 0 only if every expectation holds and every row has `exec_mode=modal_sandbox`. No verifier, invariant or timeout is changed.

Run (about 20 Modal replays, about $0.07; needs Modal auth):

    SSL_CERT_FILE=/etc/ssl/cert.pem TD_KEY=00ff PYTHONPATH=. python scripts/modal_regression.py --cap-usd 0.5

Writes `regression_results/modal_regression.{md,json}`. `--list` prints the cases without running anything.

## Recorded result

Code under test f36a0f61bd005823f71e94c02bae4d4fac52f92d, run 2026-10-19 (UTC), Python 3.12.9, modal client 1.6.2,
script spend counter $0.074. 20 of 20 expectations held, 0 violations: 5 gold accepted; 5 null and 10 incorrect repairs
rejected, all in modal_sandbox. First rejecting invariants: A null/F03 = I1 no_completed_relaunch, A F11 = I6 world_size,
B null/F16 = sample gaps and dups, B F09 = I4 ranks_inconsistent. Full table: `regression_results/modal_regression.md`.

## Not covered

Not the 115 executed tests; one seed per kind; 4 cases per kind (not the whole cheat catalog); a single run (no
repeat-run flakiness estimate). Spend is the script's estimate, not a Modal bill. This is separate from the optional macOS
native sandbox, which this suite does not exercise.
