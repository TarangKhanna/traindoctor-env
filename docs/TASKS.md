# Tasks and tiers

| kind | what happens | correct repair |
|---|---|---|
| A1 | same timeout on all ranks; a rank-dependent code path (or config entry) enters a different collective | remove the divergence |
| A2 | one rank stalled before communication (data/compute path) and never enters the collective; the launcher watchdog kills it and its peers report the timeout | fix the stall, not the timeout |
| B_SKIP | after a worker failure the job restarts and loss looks normal, but samples are skipped because checkpoint and sampler state disagree | make restore consistent |
| B_DUP | same, but samples are consumed twice | make restore consistent |
| B_OK | healthy job (control): the right answer is no fault | no repair |

**Tier A** is a single-turn bundle (capped at 3,072 Qwen3 tokens): the agent sees evidence-id-tagged records and a menu of repairs, and returns a label, cited evidence ids and a repair. It is a projection of Tier B for RL. **Tier B** is multi-turn: tools expose the same snapshot and source/config workspace, with edits applied and re-executed. Faults are executed (CPU Gloo, 3-4 ranks) and every artifact carries `source: executed | simulated`.
