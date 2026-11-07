#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_resource_stage1.sh RUN_ID [--after runs/EXISTING_CAMPAIGN]" >&2
  exit 2
fi
run_id="$1"
shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash scripts/start_campaign.sh "$run_id" --plan configs/training/resource_stage1.json "$@"
launch="runs/_launch/$run_id"
controller_pid="$(cat "$launch/pid")"
CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m pointcloud_lab.resource_monitor \
  --pid "$controller_pid" --out "$launch/resources.jsonl" --interval 10 \
  > "$launch/monitor_console.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$launch/monitor_pid"
printf 'Resource samples: %s/%s/resources.jsonl\n' "$PWD" "$launch"
