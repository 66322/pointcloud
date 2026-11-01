#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 1 || ! "$1" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Usage: bash scripts/resume_campaign.sh EXISTING_RUN_ID" >&2
  exit 2
fi
run_id="$1"
root="$PWD"
test -f "runs/$run_id/queue.json"
test -f "runs/$run_id/code/pointcloud_lab/campaign.py"
log="$root/runs/_launch/$run_id/resume_console.log"
# Use the saved code, paths and plan. An OS lock in the controller prevents
# duplicate launches. Completed/failed work is not rerun; live children adopted.
cd "runs/$run_id/code"
CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m pointcloud_lab.campaign \
  --out "$root/runs/$run_id" --resume > "$log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$root/runs/_launch/$run_id/resume_pid"
printf 'Resume controller PID: %s\nLog: %s\n' "$pid" "$log"
