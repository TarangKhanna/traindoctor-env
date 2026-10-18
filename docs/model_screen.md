# Model screens (corrected JSON-only prompt, 8 DEV episodes x G=4, 256 tok, temp 1.0; strict gate official; contended-free host, swap base ~5.6 GB)
Not a learning claim. Fence score is diagnostic only (identical to strict here: the corrected prompt removed fences).
| model | strict valid | label (valid) | V0 / V1 / V2 mean | groups spread V0/V2 | unsafe | repair ok (of valid) | peak MLX GB | tok/s |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507-4bit (chat tmpl) | 30/32 | 30x COLLECTIVE_DIVERGENCE | .25/.125/.125 | 0/0 | 4 | 4 | 4.2 | 5.9 |
| Qwen3-4B-4bit (our tmpl, think off) | 28/32 | 28x COLLECTIVE_DIVERGENCE | .25/0/0 | 0/0 | 4 | 5 | 4.2 | 6.5 |
| gemma-4-e2b-it-4bit (chat tmpl) | 0/28 (7 of 8 eps) | n/a | 0 | 0/0 | 0 | n/a | n/a | n/a |
Truth labels in the 8 episodes: CKPT_SAMPLER 3, COLLECTIVE_DIVERGENCE 2, RANK_STALLED 2, NO_FAULT 1 (x4). Every valid answer says COLLECTIVE_DIVERGENCE: V0 is 1 exactly on its two episodes, so no within-group spread.
Gemma loads in mlxrl/mlx_lm and runs, but 0 strict-valid answers (output not inspected; rows not saved); run aborted by the swap guard (+1.5 GB) after 7 episodes. Gemma template defaults to a system `<|think|>` block.
Stop criterion (V2 spread in >=2 groups and >=25% valid) met by no model.
## Training probe (Qwen3-4B-Instruct-2507, 1.9k-token prompt, G=4, checkpointing, lr=0 throwaway step, synthetic rewards)
| micro-batch | peak MLX GB | RSS GB | step s (logprob prep 35 + update 126) |
|---|---|---|---|
| 1 | 4.7 | 2.9 | 162 |
| 2 | 6.8 | 2.9 | 161 |
Swap delta negative (other processes released swap). ~22 steps/hour excluding rollouts; with ~75 s rollout per prompt, ~15 one-prompt updates/hour. Feasible in memory, slow.
Tests: tests/test_qwen_readiness.py + tests/test_first_update.py: 26 passed on the corrected prompt, no edits needed.
