# M1 finish list (blocked on owner: allowLocalBinding + runner install)

Must land before M1 is accepted. Each fix: failing test first, then the fix.

1. **Trusted paths never contain job-supplied strings** (security review, jobs/trusted.py): name
   captures by counter + content hash; keep `ckpt_id` only inside the record; assert every trusted
   write resolves under `trusted_dir(job_dir)`.
2. **Trusted process never writes through job-writable paths** (security review, jobs/launcher.py
   `samples/`, `job.json`): write run records under the trusted sibling dir, or create each
   component with O_NOFOLLOW after lstat checks. Same rule for every read of job-produced files
   by trusted code (O_NOFOLLOW walk).
3. **Sandbox opt-out** (main review, traindoctor_env/reexec.py:37): honor `TD_CI_UNSANDBOXED`
   only when `sys.platform == "linux"` and `CI` is set; every FixOutcome/episode records
   `sandboxed: bool`; any unsandboxed episode is marked invalid and logged as such.
4. Run the full suite twice (`TD_CI_UNSANDBOXED` unset on the Mac — via the tdrun `trusted`
   spec, since sandbox-exec cannot nest inside Claude Code's sandbox) + `pytest -m sweep -s`
   for family A bundle-leak numbers.
5. Exercise `jobs/sandbox.py`'s deny-by-default profile on a real torch/Gloo job; tune only the
   minimum mach-lookup / system reads; behavioural probes in tests/test_sandbox.py must run, not skip.
6. Red-on-old-code evidence for items 1, 2, 6 of the hardening pass (check out c03a981 in a
   scratch worktree, run the new tests, record failures), then green on HEAD.
7. Codex xhigh review of the execution boundary (via `tdrun codex-review`).
