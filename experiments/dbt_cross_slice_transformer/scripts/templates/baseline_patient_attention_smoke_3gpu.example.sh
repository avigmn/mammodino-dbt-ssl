#!/usr/bin/env bash
# Smoke: patient_attention MIL + 3-GPU CLS extraction. You choose LABEL + TS on artifacts-dir.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

CKPT="${CKPT:?Set CKPT=/abs/path/to/best.pt}"

TS="$(date -u +%Y%m%d_%H%M%S)"
LABEL="smoke_patient_attn_3gpu_extractMax15b_attnEp5_evalValTest"
OUT="experiments/dbt_cross_slice_transformer/runs/baseline_${LABEL}_${TS}"

python scripts/baseline_patient_pool_embeddings_dbt.py \
  --checkpoint "$CKPT" \
  --mode patient_attention \
  --cuda-devices 0,1,2 \
  --max-train-batches 15 \
  --max-val-batches 15 \
  --max-test-batches 15 \
  --eval-splits val,test \
  --attn-epochs 5 \
  --artifacts-dir "$OUT"

echo "Wrote: $OUT/metrics.json"
