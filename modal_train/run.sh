#!/usr/bin/env bash
# Usage: modal_train/run.sh <timeout_seconds> <entrypoint> [args...]   e.g. run.sh 2000 run_smoke
# Always backgrounds `modal run` (detached from the chat) with a hard wall-clock timeout, logs to modal_train/logs/,
# and lists Modal apps afterwards so orphans are visible. Never leaves a foreground wait.
set -u
T="${1:?timeout seconds}"; EP="${2:?entrypoint}"; shift 2
cd "$(dirname "$0")/.."
mkdir -p modal_train/logs
LOG="modal_train/logs/${EP}-$(date +%Y%m%d-%H%M%S).log"
nohup bash -c "timeout $T modal run modal_train/app.py::$EP $* ; echo \"exit=\$?\"; modal app list" >"$LOG" 2>&1 &
echo "pid=$! log=$LOG"
