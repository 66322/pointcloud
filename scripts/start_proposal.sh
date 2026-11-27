#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_proposal.sh RUN_ID [--with-legacy|--alignment] [--after runs/PRIOR]" >&2
  exit 2
fi
run_id="$1"
shift
plan=configs/training/proposal_20261126.json
case "${1:-}" in
  --with-legacy) plan=configs/training/fusion_with_legacy_10ep.json; shift ;;
  --alignment) plan=configs/training/proposal_alignment_pilot.json; shift ;;
esac
for arg in "$@"; do
  case "$arg" in
    --with-stf|--with-legacy|--alignment)
      echo "Choose one screening mode immediately after RUN_ID. These plans evaluate nuScenes." >&2
      exit 2 ;;
  esac
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec bash scripts/start_campaign.sh "$run_id" --plan "$plan" "$@"
