#!/bin/sh
set -eu
python - <<'PY'
from pathlib import Path
p = Path('/submission/ckpt.py')
s = p.read_text()
old = 'state.update(step=torch.tensor([step]))'
new = 'state.update(sampler=sampler.state(), step=torch.tensor([step]))'
if s.count(old) != 1:
    raise SystemExit('oracle source target is not unique')
p.write_text(s.replace(old, new, 1))
PY
