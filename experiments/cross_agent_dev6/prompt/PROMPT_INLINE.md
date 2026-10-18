You are debugging a small distributed training job (data-parallel PyTorch, Gloo, CPU, 3-4 ranks, 12 steps, checkpoint every 4 steps). The job has either failed or misbehaved in a way that matters, or it is healthy. Your task: diagnose it and, if it is faulty, repair it.

RULES
- The whole workspace is included below this message, file by file. You have no tools: do not try to read, list, run or fetch anything. Work only from the text below. Your patch is delivered only in your final message.
- Files in the workspace (embedded below):
  - logs/records.tsv : the incident evidence. One record per line: id, kind, rank, step, text. Cite records by their id (for example E1a2b3c).
  - job/train.py, job/data.py, job/ckpt.py, job/config.yaml : the editable job files.
  - job/config.py : the config parser (read-only reference; do not patch it).
- Allowed labels (exactly one):
  - COLLECTIVE_DIVERGENCE : ranks entered inconsistent collective operations.
  - RANK_STALLED_BEFORE_COLLECTIVE : one rank stalled before reaching the collective (peers wait on it).
  - CKPT_SAMPLER_INCONSISTENT : after a restart, checkpoint state and data-sampler position disagree (samples skipped or duplicated).
  - NO_FAULT : a healthy run; there is no fault, and the correct repair is to relaunch without any change (empty patch).
- A repair is judged by actually re-running the job with your patch applied: training must complete all 12 steps with every committed step consuming exactly the samples a clean reference run would, no skipped or duplicated samples, restored checkpoint/optimizer/sampler state consistent, and no more than one checkpoint interval of lost work. Do not change world_size, steps, timeouts or watchdog settings, and do not make the job exit early, skip training, or bypass checkpoint validation: such edits are rejected. Fix the root cause with the smallest correct change. Only the four editable files may be changed.
- Evidence: cite the record ids that support your diagnosis (at most 8 distinct ids; citing more is invalid). Cite only ids that appear in logs/records.tsv. Prefer the few records that discriminate this cause from the alternatives.

FINAL MESSAGE FORMAT
Your final message must contain exactly one JSON object (you may put it in a ```json fence; no other JSON objects in the message) with these keys:
{"label": "<one allowed label>",
 "evidence_ids": ["<record id>", ...],
 "fix_claim": "<one or two sentences: what you changed and why it repairs the job; or 'no change needed'>",
 "patch": "<a unified diff against the workspace files, paths like a/job/train.py and b/job/train.py, with correct @@ hunk headers and enough context lines; empty string if no change>"}
The patch string must be valid JSON (newlines as \n, quotes escaped). Keep prose outside the JSON to a minimum.
