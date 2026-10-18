# Tokenizer cap and repair-menu fix (branch modal-cuda)

## 1. Repair-menu leakage
Before (docs/audits/leakage_dev400.json on m1-audit, 200 held-out seeds): touched keys identify the label and repair
at 100% (stripped lengths 98%) against a 32.5% majority baseline.

Change (`traindoctor_env/bundle.py::make_menu`): every variant now shows the same six slots: one per repair location
(training-step sync code, per-rank sync mode, shard directory, checkpoint code), the no-change relaunch, and the
destructive rank exclusion. The slot that holds the cause carries the true repair (shown in one-line form); the other
location slots carry a wrong edit built from the job's own text by a seed-only stream (`_decoy_edits`). B_OK's correct
answer is the relaunch slot. `Candidate.edits` and the verifiers, V0/V1/V2 semantics, safety gate and strict format
gate are untouched. Items are padded with a dot filler line so stripped length is constant.

After (`python -m audit.menu_leak`, 200 held-out of 400 DEV seeds, output docs/audits/menu_leak_dev400.json): touched
keys, shapes, lengths, ids, position and n_plus/n_minus are all at or below chance (repair about 0.17-0.21 vs 0.325;
label 0.325 = majority). Test: `tests/test_menu_leak.py` (240 seeds, pure Python).

Direction leak, closed (second change): a repair-location item now renders BOTH forms of the location (the correct
code and the faulty code), each line marked `~`, separated by `~ ---`, in an order fixed by a hash of the candidate id
(never by which form is the job's current one), with context lines from the CLEAN job text (the buggy job with the true
repair applied). The sync location always shows the one-line conditional form; the checkpoint location always shows
both its write and restore sites. True repair and decoy therefore have the same shape and context; the faulty location
has to be read from the incident records. `Candidate.edits` (what is executed) is unchanged; `Candidate.shown` is
display only, so the sync item for A1 displays the one-line faulty form rather than the literal `if False:` edit.

Before/after, 400 DEV seeds (200 held-out for lookup, 5-fold across seeds for the regression), produced by
`docs/audits/menu_leak_before_after.py` -> `docs/audits/menu_leak_before_after.json` (Wilson 95% CI, chance 0.325/0.36):
- lookup on added/removed diff text: 0.740 [0.675, 0.796] -> 0.175 [0.129, 0.234]
- lookup on first form / unordered variant set: 0.175 -> 0.325 [0.264, 0.393] (chance)
- character 4-gram logistic regression on menu text, cross-validated over seeds: 0.530 [0.481, 0.578] -> 0.300
  [0.257, 0.347] (chance 0.36). It detects the old leak, so it can fail.
`tests/test_menu_leak.py` (240 seeds) now asserts all of these at chance; the strict xfail is removed.

## 2. Tokenizer cap
Before (docs/token_audit.json): true Qwen3 token counts of bundle prompts median 2367, max 2771 (> 2048).

Change: `est_tokens` is now a deterministic upper bound of the Qwen3 count (no tokenizer needed at generation time, so
bundles stay byte-identical across machines); `bundle.tokens` = bound + `PROMPT_OVERHEAD` (160; the real frame is 63).
`make_bundle` compacts when over the cap: non-certificate records are dropped, noise kinds first and the incident
summary last, each group in the order of a stream seeded by the scenario seed alone; the certificate is never dropped
(BundleTooLarge if it alone does not fit). To make room: K_MENU stays 6 slots, ITEM_CHARS 640 -> 360, record QUOTA
log/trace/samples 6/8/4 -> 5/6/3.

Verified (pure Python, `tests/test_token_cap.py`, offline Qwen3-0.6B tokenizer): bound >= real count on synthetic
record chunks (worst real/bound 0.98 at 1.08 safety; now 1.10), menu diffs and padded configs of 30 scenarios; frame
<= 160 tokens. Menu plus config is about 800 + 160 real tokens of the 2048.

## 3. Test edits
`tests/test_tiera.py`, `test_scenarios.py`, `test_bundle_sweep.py`, `test_qwen_readiness.py`: menu roles are now
correct / noop / exclude_rank / decoy_*; equality of menu edits to the reference edits replaced by the re-execution
table (the menu shows one-line repairs); `bundle.tokens` includes the overhead; decoys must fail by re-execution.
Golden prompt fixtures unchanged (they cover the frame, not the bundle).

## NOT verified (needs executed snapshots / Gloo, not run here)
- The neutral display is only a rendering: whether a model can still pick the faulty location from the records alone
  (oracle-solves-from-bundle with a bundle-evidence-only chooser) is not tested; the existing oracle test still selects
  the item by role, so it does not exercise the new display. The model is not told which form is current.
- True-token size of executed bundles with the new items (menu items stay exactly ITEM_CHARS; the est_tokens bound
  tests on synthetic menus pass).
- Real-token size of executed bundles (`test_executed_bundle_prompt_is_within_the_cap_in_true_tokens`), and that no
  seed raises BundleTooLarge (`test_bundle_sweep`): the certificate plus incident summary must fit in about 500
  bound-tokens of record space.
- Decoy outcomes by re-execution (every decoy must fail; perrank `reduce_bcast` on B_OK is the one I doubt).
- Oracle-solves-from-bundle on every variant (test_tiera); A1 shown as `if False:` repair must pass re-execution.
- Required next: DEV rebaseline and SFT regeneration (menu and prompts changed, old data invalid).
