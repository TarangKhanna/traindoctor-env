# Repair checkpoint recovery

A small two-rank CPU/Gloo training job completes after preemption, but its resumed
training computation is inconsistent. Investigate the checkpoint code, training
configuration and available job records. Repair the underlying checkpoint behavior
without changing the training mathematics or dropping committed work.

Your submission is **`/submission/ckpt.py`**. Edit that real Python source file.
The other job files and input layout are in `/workspace/incident/job/` and
`/workspace/incident/data/`. Do not submit a diagnosis label or select a menu item.

You can inspect source and run a fresh diagnostic preemption/restart using the
existing launcher. Diagnostic results inside your workspace are untrusted; final
scoring independently re-executes your submitted source in a separate verifier.
This smoke task tests whether repaired code produces recoverable checkpoints;
it does not require repairing already-lost state inside an old checkpoint.

```bash
python - <<'PY'
from pathlib import Path
import tempfile, shutil
from jobs.config import loads
from jobs.build import materialize
from jobs.resubmit import resubmit
root = Path(tempfile.mkdtemp(prefix="td-diagnostic-")) / "jobdir"
cfg = loads(Path("/workspace/incident/job/config.yaml").read_text())
materialize(root, cfg)
shutil.copyfile("/submission/ckpt.py", root / "job/ckpt.py")
r = resubmit(root, kill=(0, 9), restart_on_failure=True, allow_unsandboxed=True)
print({"status": r.status, "error_class": r.error_class, "records": str(root)})
PY
```

A clean exit is insufficient: retain intended sample order, optimizer state and
checkpoint continuity. Agent-side diagnostics are not the verifier reward. Leave
`/submission/` containing only the submitted regular `ckpt.py` file. Finish with a
brief explanation of your change and the execution evidence you inspected.
