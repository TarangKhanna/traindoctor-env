# Local learning log

Append-only record of verified findings, corrections and limits. These are newly
observed project findings, not claims of global novelty. Historical evidence and
earlier protocols retain their original meanings.

## 2026-10-29 — One state-only recovery demonstration

- **Actual continuation matters.** Historical Task B verifies repaired source by
  constructing a fresh failed job. That cannot by itself establish that a selected
  checkpoint from an existing incident was recovered correctly. The separate
  `state_recovery_v1` protocol preserves genuine captures and grades the chosen
  continuation. Evidence and scope: [protocol contract](STATE_RECOVERY_V1.md).
- **Byte integrity is insufficient.** Checksummed artifacts with incompatible
  sampler or momentum state finish with the wrong trajectory. All twelve tested
  continuations complete; selective repairs reverse validity across the two states.
  This proves the constructed mechanism, not its prevalence in production.
- **A robust optimum survives.** Full committed restoration is correct and optimal
  in both cases even with explicit recomputation cost. Scratch is correct but more
  costly. Keep both; the result supports an engineering demo, not a difficult
  conditional-selection RL task. See the companion research report and raw receipts.
- **Preserve anchor provenance.** Trusted committed captures are original execution
  artifacts; inconsistent delivered checkpoints are separate derived artifacts.
  The data schedule must preserve the committed prefix and use the same continuation
  challenge for every compared intervention and reference.
- **Manifest updates are part of restoration.** Review caught a stale full-restore
  digest in a preliminary execution. Updating the manifest and validating each
  intervention fixed it. Preliminary evidence was retained; it is not the final run.
- **Loopback environment variables are not binding evidence.** Live `lsof` exposed
  a wildcard TCPStore listener despite loopback `MASTER_ADDR`. A pre-bound loopback
  socket passed through PyTorch's existing TCPStore API fixed the bind and avoided
  a free-port race. The live listener regression passed, 241 focused tests passed,
  and all twelve continuations/recount passed again with unchanged outcomes.
- **Do not build a code-execution security framework for one demo.** Review found
  bypasses in an exploratory generic validator. It was discarded. The companion
  code-repair replay executes only one exact, manually reviewed proposal under the
  existing sandbox. This is not a general safe executor for model output.

Verification commands and exact execution configuration are in the companion
research report, `experiments/state_recovery_v1/REPORT.md`. Linux execution and
independent external reproduction remain unverified. The existing sandbox contract
must not be described as stronger than its measured platform/backend guarantees.

2026-10-29 README compression: lead with checksum-valid inconsistent recovery; move the previous detailed landing page intact (relative links rebased) to docs/PROJECT_DETAIL.md. Retain historical evidence and numerical checks. Critic caught that structural well-formedness does not imply checksum validation; wording now explicitly describes recomputed manifest checksums. Documentation changes only.

2026-10-30 Harbor catches: separate hidden-test delivery is not a scoring trust boundary. Use a fresh verifier with source-only transfer, root-protected runtime/records and actual unprivileged-worker proof. Review caught missing bridge image copy, required root KILL/SETPCAP, hardlink ownership hazards and shared verifier-log inode tampering. Fixed before any container execution; unit/static proof is not operational isolation proof. Harbor imports dotenv automatically: CLI must run from a clean temporary directory with supported dotenv disabling. CLI lint is offline; check is LLM-backed and remains forbidden here. Exact source/version and limitations: docs/HARBOR.md.

2026-10-30 — Actual Harbor execution exposed compatibility issues that static adapter tests missed: Docker backend rejected provider no-network, read-only roots blocked transport directories, and Docker Desktop could read a tmpfs file through exec while docker-cp could not copy it. Named transport volumes preserve source-only handoff and separate trusted scoring. Interface-name equality also falsely rejected dormant kernel tunnels; checking UP flags plus absent NET_ADMIN and external denial distinguishes a real network path. AI diff review caught Harbor's reward.txt fallback: clearing only reward.json can retain stale success on infrastructure failure. Both formats are now cleared and regression-tested. All four final controls and native verdict/source comparisons pass; no benchmark or security-generalization claim follows.

Session learning: reuse the native Harbor implementation and inspect its real transfer/reward behavior before changing the adapter. The near-miss was treating CLI exit 0 or static tests as a completed trial; only trial exceptions and verifier receipts establish the result. A compact Docker transport discriminator and explicit retained-container option would have shortened diagnosis. No new closure/scheduler framework was added.
