#!/usr/bin/env bash
# Smoke test: 3-GPU embedding extraction (DataParallel) + short Transformer train.
# Copy → chmod +x → set CKPT. Timestamp is appended by Python (UTC), not here.

set -euo pipefail

SANDBOX="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "$SANDBOX/../.." && pwd)"
cd "$REPO"

CKPT="${CKPT:?Set CKPT=/abs/path/to/best.pt}"

# Descriptive slug only — NO manual timestamp (added automatically as ..._<UTC_TS>/).
RUN_NAME="smoke_3gpu_extractMax10b_tfMax3ep_earlyStopPatience99_evalVal"

python experiments/dbt_cross_slice_transformer/scripts/run_cross_slice_transformer.py \
  --checkpoint "$CKPT" \
  --cuda-devices 0,1,2 \
  --max-train-batches 10 \
  --max-val-batches 10 \
  --epochs-max 3 \
  --early-stop-patience 99 \
  --eval-splits val \
  --verbose-epoch-metrics \
  --run-name "$RUN_NAME" \
  --runs-root experiments/dbt_cross_slice_transformer/runs

echo "Done. New folder: experiments/dbt_cross_slice_transformer/runs/cross_slice_transformer_${RUN_NAME}_<UTC_stamp>/"
