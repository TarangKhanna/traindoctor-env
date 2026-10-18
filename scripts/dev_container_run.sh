#!/bin/bash
# DEV mode: run ONE fixed command of repo-owned, reviewed code in a minimal Linux CPU container.
#   dev_container_run.sh prepare | tests_executed | token_cap | decoys | audit_live
# Run it from a checkout; the fixed command names keep flags, mounts and the key out of the caller's hands. Not for agent-edited/untrusted code (UNTRUSTED mode is separate).
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 prepare|tests_executed|token_cap|decoys|audit_live" >&2; exit 64; }
WT=${TD_DEV_WT:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=td-dev:1
DOCKER=
for d in /usr/local/bin/docker /opt/homebrew/bin/docker /Applications/Docker.app/Contents/Resources/bin/docker; do [ -x "$d" ] && { DOCKER=$d; break; }; done
[ -n "$DOCKER" ] || { echo "docker not found" >&2; exit 69; }
case "$1" in
  prepare)        CMD='python -m modal_train.prepare --dev 48 --train 64' ;;
  tests_executed) CMD='python -m pytest -q -rs -p no:cacheprovider -c pytest.ini tests/' ;;
  token_cap)      CMD='python -m pytest -q -rs -p no:cacheprovider -c pytest.ini tests/test_token_cap.py' ;;
  decoys)         CMD='python -m pytest -q -rs -p no:cacheprovider -c pytest.ini tests/test_scenarios.py tests/test_tiera.py' ;;
  audit_live)     CMD='test -f audit/live.py || { echo "audit/live.py missing: merge m1-audit first" >&2; exit 66; }
                       python -m audit.report leakage --out audit_out/report &&
                       python -m audit.live --out audit_out --per-kind 4 --workers 1 &&
                       python -m audit.report all --in audit_out --out audit_out/report' ;;
  *) echo "unknown command $1" >&2; exit 64 ;;
esac
[ -d "$WT" ] && [ ! -L "$WT" ] || { echo "worktree $WT missing or a symlink" >&2; exit 66; }
if [ "$1" = audit_live ] && pgrep -fl 'mlx_lm|sglang|mlxrl' >/dev/null; then echo "a model process is running; refusing to share the CPU" >&2; exit 75; fi
# fresh scratch dir
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/scratch.XXXXXX")
chmod 777 "$SCRATCH"
echo "[dev-container] command=$1 worktree=$WT scratch=$SCRATCH"
INNER='set -e
export HOME=/scratch/home PYTHONPATH=/scratch/w CI=1 TD_CI_UNSANDBOXED=1
export TD_KEY=$(python -c "import os;print(os.urandom(16).hex())")
mkdir -p /scratch/home /scratch/w
tar -C /work --exclude=.git -cf - . | tar -C /scratch/w -xf -
cd /scratch/w
'"$CMD"
exec "$DOCKER" run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 512 --memory 8g --cpus 4 --shm-size 512m --user 65534:65534 \
  -v "$WT":/work:ro -v "$SCRATCH":/scratch:rw --tmpfs /tmp:rw,size=1g,mode=1777 \
  "$IMAGE" bash -c "$INNER"
