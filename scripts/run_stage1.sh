#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_stage1.sh preflight|diagnose-pilot|diagnose-full RUN_ID" >&2
  exit 2
fi
phase="$1"
run_id="$2"
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "RUN_ID may contain only letters, digits, underscore and hyphen" >&2
  exit 2
fi
paths="configs/paths.local.json"
out="runs/$run_id"
case "$phase" in
  preflight)
    python -m pointcloud_lab.preflight --paths "$paths" --out "$out"
    ;;
  diagnose-pilot)
    python -m pointcloud_lab.diagnose --paths "$paths" --out "$out" \
      --limit 100 --confirm-original-point-order
    ;;
  diagnose-full)
    python -m pointcloud_lab.diagnose --paths "$paths" --out "$out" \
      --full --confirm-original-point-order
    ;;
  *)
    echo "Unknown phase: $phase" >&2
    exit 2
    ;;
esac
