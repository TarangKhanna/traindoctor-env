#!/usr/bin/env bash
# Usage: modal_train/run_cpu.sh <timeout_s> <entrypoint> [args]; backgrounded, logged, lists apps afterwards.
set -u
T="${1:?}"; EP="${2:?}"; shift 2
cd "$(dirname "$0")/.."
M=${MODAL_BIN:-modal}
mkdir -p modal_train/logs
LOG="modal_train/logs/${EP}-$(date +%H%M%S).log"
nohup bash -c "timeout $T $M run modal_train/cpu_app.py::$EP $*; echo exit=\$?; $M app list" >"$LOG" 2>&1 &
echo "pid=$! log=$LOG"
