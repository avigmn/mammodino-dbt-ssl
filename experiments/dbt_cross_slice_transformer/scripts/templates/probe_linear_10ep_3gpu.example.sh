#!/usr/bin/env bash
# Linear probe on slices: torchrun 3 GPUs. --run-name is descriptive; Python appends local _stamp.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

CKPT="${CKPT:?Set CKPT=/abs/path/to/best.pt}"

torchrun --standalone --nproc_per_node=3 scripts/train_linear_probe_dbt.py \
  --num-gpus 3 \
  --device cuda \
  --checkpoint "$CKPT" \
  --epochs 10 \
  --run-group manual_runs \
  --run-name probe_linear_3gpu_10ep_smoke_smallbatch \
  "$@"

echo "Artifacts under experiments/dbt_dino_runs/probe/manual_runs/probe_linear_3gpu_10ep_smoke_smallbatch_<stamp>/"
