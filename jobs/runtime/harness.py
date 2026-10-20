"""Entry point of every rank: `python jobs/runtime/harness.py` (cwd = the job directory).

It only reports, after each optimizer step, a step counter and a state digest to the launcher. That is
EVIDENCE for the agent-readable run records and nothing else: it runs inside the job's process, so the job can
silence or forge it, and no verdict depends on it. What decides a verdict is outside this process: the data
server's log, the checkpoints the launcher copies when they are committed, and the launcher's process history."""
import hashlib
import os
import runpy
import sys

from torch.optim.optimizer import register_optimizer_step_post_hook

import td_client


def state_digest(opt) -> str:
    """Parameters and momentum buffers, in optimizer order."""
    h = hashlib.sha256()
    for g in opt.param_groups:
        for p in g["params"]:
            h.update(p.detach().contiguous().numpy().tobytes())
            buf = opt.state.get(p, {}).get("momentum_buffer")
            h.update(buf.detach().contiguous().numpy().tobytes() if buf is not None else b"-")
    return h.hexdigest()


def main() -> None:
    sys.path.insert(0, os.path.abspath("job"))
    steps = [0]

    def post(opt, args, kwargs):
        steps[0] += 1
        td_client.notify(dict(op="step", n=steps[0], digest=state_digest(opt)))

    register_optimizer_step_post_hook(post)
    sys.argv = ["job/train.py"]
    runpy.run_path("job/train.py", run_name="__main__")


if __name__ == "__main__":
    main()
