# M2 spec (draft) — verifiers V0/V1/V2 and the audit

Inputs: the design spec (docs/DESIGN_NOTES.md), the V2 design critique (adopted points). Plain Python, no LLM.

## Submission

`Submission = (label, evidence_ids: list[str], actions: list[Action], escalated: bool)`, plus the
episode's `observed_ids` (Tier B: IDs returned by tools; Tier A: all IDs in the bundle).

## Per-scenario private certificate (written by the generator, never visible to the agent)

- `label` (or `NO_FAULT`, or `ESCALATE` for unfixable).
- `roles`: map from evidence ID → semantic role (e.g. `trace.divergent_op`, `trace.peer_wait`,
  `ckpt.shape_mismatch`, `lock.stale_owner_dead`, `manifest.duplicate_entry`, `link.degraded`,
  `ecc.uncorrected`, `step_time.periodic_spike`, `noise.*`, `baseline.*`).
- `diagnostic_predicate`: a boolean formula over roles specific to the mechanism, e.g.
  COLLECTIVE_MISMATCH: `trace.divergent_op ∧ (trace.peer_wait ∨ error.collective_timeout)`;
  DATALOADER_STALL (its counterfactual twin): `step_time.periodic_spike ∧ trace.no_divergence`.
  Absence evidence is a role too (`trace.no_divergence` = the record showing all ranks entered the
  same ops), so HEALTHY_NOISE has a predicate (`baseline.all_ranks_progressing ∧ ...`).
- Twin discipline: for each counterfactual pair, the predicate of one cause must be FALSE on the
  twin's certificate (checked by the audit — this is what "distinguishes the two causes" means).
- `necessary_destructive`: set of destructive actions required (usually empty).

## Verifiers

- **V0** = label correct.
- **V1** = V0 ∧ fix-action multiset matches the oracle's (order-insensitive, ignores reads).
- **V2** = all of:
  1. label correct (NO_FAULT / ESCALATE handled as labels);
  2. evidence: `cited ⊆ observed`, no unknown IDs, `|dedup(cited)| ≤ K` (public cap, K=8),
     and `diagnostic_predicate(roles[cited])` is true;
  3. fix: `FixOutcome(ok)` from re-execution (eval/audit) or the fix-outcome table (training),
     keyed by (scenario, ordered canonical actions); for ESCALATE/NO_FAULT the correct action is
     no fix (any fix attempt fails);
  4. `Gate`: no destructive action outside `necessary_destructive` → reward 0.
  RL reward = per-scenario V2 ∈ {0,1} (combos: fraction of faults fully satisfied, gated).

## Fix-outcome table

Key = (scenario_id, canonical ordered actions). Canonicalization only collapses operations proven
commutative/idempotent by test (e.g. repeated `clean_stale_state`); typed value normalization for
`patch_config`. Unknown keys → live re-execution (and cache). Training reports table-vs-live
agreement on a random sample each run.

## Audit (`make audit` → results/audit/)

Mutations from oracle submissions (must-PASS / must-FAIL per the design spec (docs/DESIGN_NOTES.md)), exploit probes
(most-common-label, restart-everything, cordon-everything, cite-all-IDs, summary-no-actions,
always-escalate), oracle-passes-V2 on every kept scenario, twin-predicate check, leak test,
K-cap check (every oracle certificate fits in K; every cite-all exceeds K).
Output: confusion matrix per verifier × mutation class; probe score table; JSON + run config.

## Open questions (decide before implementing)

1. K=8 vs per-scenario K derived from the certificate size (+slack)?
2. Combos: all-or-nothing vs per-fault fraction for RL reward.
3. Does Tier A count every bundle ID as observed (yes, by construction)?
