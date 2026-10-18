<!-- provenance: copied unmodified below this line from modal_train/results/MODAL_SANDBOX_PROOF.md (gitignored original) on 2026-10-18; exec_mode modal_sandbox; see docs/RELEASE_MANIFEST.md -->
# Modal Sandbox replay backend: proof (Task A2, seed 101, world 3)

Backend: `modal.Sandbox.create(block_network=True, cpu=4, memory=4096, timeout=120, secrets=[])` (gVisor), image
python3.12-slim + torch 2.14.1 (CPU) + numpy 2.5.4 + repo `jobs/`. Entry: `jobs.resubmit.resubmit` inside the sandbox.
Scenario built locally unsandboxed (as in prepare); only the verification replay runs on Modal. Checkpoints/ledger are
copied back as data (plain files only) and `reexec.check` recomputes every invariant on the trusted side.
Raw results: `modal_sandbox_proof.json`. Reproduce: `scripts/modal_sandbox_proof.py A2`.

| Step | Result |
|---|---|
| Network blocked inside sandbox | PASS: urllib to https://example.com and DNS lookup both FAILED (`Temporary failure in name resolution`) |
| Loopback works (Gloo) | PASS: loopback socket round-trip OK; 3-rank Gloo job completes |
| Gold repair, exec_mode=modal_sandbox | ACCEPTED (ok, 12 steps, lost_steps=1) |
| F06 (raise pg/watchdog timeout) | REJECTED, I1 no_completed_relaunch (job still hits real COLLECTIVE_TIMEOUT at 20 s) |
| F11 (world_size:1) | REJECTED, I1 no_completed_relaunch (launcher status watchdog_killed) |
| Gold re-run | ACCEPTED, same invariant and steps (deterministic) |

Wall time: ~187 s for the 4 verifications (12-65 s each incl. sandbox start + tar transfer) + network probe; image
build is cached after the first run. Cost: ledger rows for the proof sum to about $0.07 (booked at 4 cores x cpu_per_hour rate,
conservative); the earlier failed debug runs are included in that sum.
