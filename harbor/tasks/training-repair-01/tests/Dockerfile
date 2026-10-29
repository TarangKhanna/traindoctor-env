# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/traindoctor TD_KEY=harbor-local-smoke-key
WORKDIR /opt/traindoctor
RUN command -v setpriv
COPY pyproject.toml ./
COPY jobs/ ./jobs/
COPY traindoctor_env/ ./traindoctor_env/
RUN python - <<'PY'
import subprocess, sys, tomllib
from pathlib import Path
requirements = tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies']
torch = [r for r in requirements if r.startswith('torch==')]
if len(torch) != 1:
    raise SystemExit('require one exact project Torch pin; no fallback')
others = [r for r in requirements if r not in torch]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                       '--index-url', 'https://download.pytorch.org/whl/cpu', *torch])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                       '--index-url', 'https://pypi.org/simple', *others])
PY
RUN python - <<'PY'
from pathlib import Path
from traindoctor_env.scenario import Scenario
from traindoctor_env.snapshot import execute
private = Path('/opt/traindoctor-fixture')
execute(Scenario(seed=777, kind='B_DUP', variant=0, world_size=2, job_seed=777), private)
private.chmod(0o700)
PY
COPY harbor/tasks/training-repair-01/tests/grade.py /tests/grade.py
COPY harbor/tasks/training-repair-01/tests/worker_bridge.py /tests/worker_bridge.py
COPY --chmod=755 harbor/tasks/training-repair-01/tests/test.sh /tests/test.sh
RUN mkdir -p /submission /logs/verifier /logs/artifacts /logs/agent \
    && chmod 0755 /logs /logs/agent \
    && chmod 0700 /submission /logs/verifier /logs/artifacts
USER 0
WORKDIR /work
