#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/start_view_experiments.sh DESCRIPTIVE_RUN_ID" >&2
  exit 2
fi
run_id="$1"
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "RUN_ID may contain only letters, digits, underscore and hyphen" >&2
  exit 2
fi
out="runs/$run_id"
launch="runs/_launch/$run_id"
if [[ -e "$out" || -e "$launch" ]]; then
  echo "Existing run/launch directory: choose a NEW run ID. Existing work is preserved." >&2
  exit 2
fi
test -f configs/paths.local.json
mkdir -p runs/_launch
mkdir "$launch"
nohup python -u -m pointcloud_lab.run_views \
  --paths configs/paths.local.json --out "$out" \
  > "$launch/console.log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$launch/pid"
printf 'Started batch PID: %s\n' "$pid"
printf 'Batch log: %s/%s/console.log\n' "$PWD" "$launch"
printf 'Outputs: %s/%s/\n' "$PWD" "$out"
printf 'Progress: tail -f %s/console.log\n' "$launch"
