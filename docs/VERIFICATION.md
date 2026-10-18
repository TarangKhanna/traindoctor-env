# Verification: invariants, trust boundary, red-team details

Moved from the top-level README (the short, illustrated version is in the README itself).

Many wrong repairs make the job stop failing. A fake repair that collapses the job to one process (`world_size: 1`, red-team case F11) also ends cleanly and completes its steps, yet the same quickstart run shows it broke the computation: `sample_gaps=96`, parameters and states differ from the uninterrupted reference. Other fakes in the red-team suite swallow the exception, raise timeouts, drop a rank, patch `torch.distributed`, or forge the job's own logs. The verifier therefore checks, at the committed optimizer-step boundary: samples consumed per step equal the reference schedule (none skipped, none in two committed updates), restored model/optimizer/sampler state equals what the checkpoint committed, lost work is within a checkpoint interval, all ranks end in the same state, and protected files were not tampered with.

Verifier ladder: V0 label correct; V1 adds cited evidence that was observed and satisfies the mechanism's evidence predicate; V2 adds a verified repair by re-execution. Components are reported separately, and claimed-vs-verified is logged for every episode.

| `exec_mode` | what runs the repair | counts as verified? |
|---|---|---|
| `modal_sandbox` | disposable Modal Sandbox (gVisor, no network, no credentials); invariants recomputed outside it | yes: the intended path for untrusted repairs |
| `sandboxed` | local macOS `sandbox-exec` profile (`jobs/sandbox.py`); documented residual: loopback stays open for Gloo | development only |
| `dev_container` | trusted pinned code in local Docker | no: exploratory training reward only, never counted by strict V2 |
| unsandboxed local | no isolation | no: forced `ok=False` (INVALID) |

Eval and audit code must never reach the `dev_container` reward. Design rationale: [`docs/DESIGN_NOTES.md`](DESIGN_NOTES.md). Reporting security issues: [`SECURITY.md`](../SECURITY.md).

### Red-team details

The 38 execution failures show that those cheats do not run; invariant coverage is claimed for the 48 only. The one false reject among the 15 legitimate repairs is B_DUP alt2 (sampler position recomputed in `restore()`), failing only a checkpoint-format check. Verifier v1.1 accepts it and still accepts 0/85 verified cheats; it is prepared but not active ([`docs/VERIFIER_V11.md`](VERIFIER_V11.md)). A borderline surface patch (A2: skip the shard wait) is accepted by v1 as outcome-equivalent and motivated the A2v2 scenario.
