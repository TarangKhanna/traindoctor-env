# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/traindoctor TD_KEY=harbor-local-smoke-key
WORKDIR /opt/traindoctor
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
RUN useradd --create-home --uid 1000 agent
RUN python - <<'PY'
from pathlib import Path
import shutil
from traindoctor_env.scenario import Scenario
from traindoctor_env.snapshot import execute
private = Path('/opt/traindoctor-fixture')
execute(Scenario(seed=777, kind='B_DUP', variant=0, world_size=2, job_seed=777), private)
private.chmod(0o700)
public = Path('/workspace/incident')
shutil.copytree(private/'fixture', public, ignore=shutil.ignore_patterns('_private', '__pycache__'))
shutil.copytree(private/'public', Path('/workspace/records'))
shutil.rmtree(private)
PY
COPY --chmod=755 <<'SH' /usr/local/bin/prepare-submission
#!/bin/sh
set -eu
if [ ! -e /submission/ckpt.py ]; then
    cp /workspace/incident/job/ckpt.py /submission/ckpt.py
fi
exec "$@"
SH
RUN mkdir -p /submission /logs/agent /logs/verifier /logs/artifacts /solution \
    && chown 1000:1000 /submission /logs/agent /logs/artifacts \
    && chmod 0700 /submission /logs/verifier \
    && chmod 0755 /logs /logs/agent /logs/artifacts /solution
USER 1000
WORKDIR /submission
ENTRYPOINT ["/usr/local/bin/prepare-submission"]
CMD ["sleep", "infinity"]
