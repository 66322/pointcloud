#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_training_experiments.sh RUN_ID [run_training options]" >&2
  exit 2
fi
run_id="$1"
shift
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Invalid run ID" >&2
  exit 2
fi
launch="runs/_launch/$run_id"
if [[ -e "runs/$run_id" || -e "$launch" ]]; then
  echo "Existing run/launch directory: choose a fresh run ID" >&2
  exit 2
fi
test -f configs/paths.local.json
mkdir -p runs/_launch
mkdir "$launch"
CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m pointcloud_lab.run_training \
  --paths configs/paths.local.json --out "runs/$run_id" "$@" \
  > "$launch/console.log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$launch/pid"
printf 'Training queue PID: %s\nBatch log: %s/%s/console.log\nOutputs: %s/runs/%s/\n' \
  "$pid" "$PWD" "$launch" "$PWD" "$run_id"
