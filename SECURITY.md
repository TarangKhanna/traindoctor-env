# Security policy

## Trust boundary
TrainDoctor executes code proposed by an agent (a "repair") to check whether a failed training job recovers. Treat every repair as untrusted.

- Untrusted repairs must run only in a disposable sandbox: the `modal_sandbox` backend (network-less gVisor Modal Sandbox, no credentials, invariants recomputed outside the sandbox) is the intended path. Results from `dev_container` (local Docker) or from local unsandboxed runs are for development and training reward only; the strict verifier refuses to count them as verified recovery.
- The local macOS `sandbox-exec` profile in `jobs/sandbox.py` has a documented residual: loopback ports stay open for Gloo, so other loopback services on the same machine are reachable from a job. Do not run untrusted repairs on a machine with sensitive loopback services.
- Never run an agent-proposed diff outside a sandbox just to "see if it works".

## Reporting
Please report suspected verifier bypasses (a repair that is accepted but should not be), sandbox escapes, or secret exposure through a GitHub private security advisory on this repository (Security tab, "Report a vulnerability"). Do not open a public issue for these. Verifier bypasses that are reproducible against `scripts/redteam_suite.py` cases are especially welcome; accepted reports become new red-team cases.

This is research code without an external security review.
