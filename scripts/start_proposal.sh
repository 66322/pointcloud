#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_proposal.sh RUN_ID [--seed N] [--with-stf] [--after runs/PRIOR]" >&2
  exit 2
fi
run_id="$1"
shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec bash scripts/start_campaign.sh "$run_id" --plan configs/training/proposal_20261126.json "$@"
