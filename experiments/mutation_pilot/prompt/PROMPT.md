You are debugging a small distributed training job (data-parallel PyTorch, Gloo, CPU, 3-4 ranks, 12 steps, checkpoint every 4 steps). The job has a bug that makes it fail or recover incorrectly when a worker fails and the scheduler restarts it, or that makes its ranks fail at a collective. Your task: diagnose the bug and repair it.

RULES
- Work only inside the workspace directory named in the last line of this message (WORKSPACE_DIR). Read only files under that directory. Do not read, list, or search anything outside it. You cannot run the job; you can only read files and reason.
- Files in the workspace:
  - logs/records.tsv : the incident evidence. One record per line: id, kind, rank, step, text. Cite records by their id (for example E1a2b3c).
  - job/train.py, job/data.py, job/ckpt.py, job/config.yaml : the editable job files.
  - job/config.py : the config parser (read-only reference; do not patch it).
- A repair is judged by actually re-running the job with your patch applied under the same incident (the same rank failure and automatic restart, or the same relaunch from the latest valid checkpoint): training must complete all 12 steps with every committed step consuming exactly the samples a clean reference run would, no skipped or duplicated samples, restored checkpoint/optimizer/sampler state consistent with what the checkpoint committed, ranks consistent, and no more than one checkpoint interval (4 steps) of lost work. Do not change world_size, steps, timeouts or watchdog settings, and do not make the job exit early, skip training, or bypass checkpoint validation: such edits are rejected. Fix the root cause with the smallest correct change. Only the four editable files may be changed.
- Evidence: cite the record ids that support your diagnosis (at most 8 distinct ids; citing more is invalid). Cite only ids that appear in logs/records.tsv.

FINAL MESSAGE FORMAT
Your final message must contain exactly one JSON object (you may put it in a ```json fence; no other JSON objects in the message) with these keys:
{"label": "<your root-cause category in at most 12 words>",
 "evidence_ids": ["<record id>", ...],
 "fix_claim": "<one or two sentences: what you changed and why it repairs the job>",
 "patch": "<a unified diff against the workspace files, paths like a/job/train.py and b/job/train.py, with correct @@ hunk headers and enough context lines>"}
The patch string must be valid JSON (newlines as \n, quotes escaped). Keep prose outside the JSON to a minimum.
