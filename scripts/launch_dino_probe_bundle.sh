#!/usr/bin/env bash
# One folder tree per run: full DINO SSL training + downstream probes (no MIL training here).
# Layout: experiments/dino_probe_bundles/run_<timestamp>/<run_name>_<ts>/...
#
# Steps:
#   1) DINO train (default 100 epochs, early stopping patience 5) → checkpoints/, logs/
#   2) Frozen patient-attention baseline (val+test) → probe_frozen_patient_attention/
#   3) Linear slice-head probe (val+test, confusion PNG/JSON) → probe_linear_slice_head/
#
# Usage:
#   chmod +x scripts/launch_dino_probe_bundle.sh
#   ./scripts/launch_dino_probe_bundle.sh               # 3 GPUs (torchrun)
#   NUM_GPUS=1 ./scripts/launch_dino_probe_bundle.sh   # single GPU
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO}"

PYTHON="${PYTHON:-${REPO}/.venv_py312_clean/bin/python}"
NUM_GPUS="${NUM_GPUS:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
BUNDLE_PARENT="${REPO}/experiments/dino_probe_bundles"
BUNDLE="${BUNDLE_PARENT}/run_${TS}"
mkdir -p "${BUNDLE}"

RUN_NAME="${RUN_NAME:-dino_fulltrain_100ep_es5}"
EPOCHS="${EPOCHS:-100}"
ES_PATIENCE="${ES_PATIENCE:-5}"
DATA_ROOT="${DATA_ROOT:-${REPO}/../dbt_simclr_project}"

echo "Bundle parent: ${BUNDLE}"
echo "Training DINO: epochs=${EPOCHS} early_stopping_patience=${ES_PATIENCE} num_gpus=${NUM_GPUS}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${REPO}/scripts/train_dino.py" \
    --config configs/dino_dbt.yaml \
    --device cuda \
    --num-gpus "${NUM_GPUS}" \
    --epochs "${EPOCHS}" \
    --early-stopping-patience "${ES_PATIENCE}" \
    --bundle-root "${BUNDLE}" \
    --run-name "${RUN_NAME}" \
    2>&1 | tee "${BUNDLE}/dino_train.log"
else
  "${PYTHON}" "${REPO}/scripts/train_dino.py" \
    --config configs/dino_dbt.yaml \
    --device cuda \
    --epochs "${EPOCHS}" \
    --early-stopping-patience "${ES_PATIENCE}" \
    --bundle-root "${BUNDLE}" \
    --run-name "${RUN_NAME}" \
    2>&1 | tee "${BUNDLE}/dino_train.log"
fi

RUN_DIR="$(find "${BUNDLE}" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
CKPT="${RUN_DIR}/checkpoints/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found at ${CKPT}"
  exit 1
fi

echo "RUN_DIR=${RUN_DIR}"
echo "CKPT=${CKPT}"

ATTN_DIR="${RUN_DIR}/probe_frozen_patient_attention"
mkdir -p "${ATTN_DIR}"
POOL_EXTRA=()
if [[ "${NUM_GPUS}" -ge 3 ]]; then
  POOL_EXTRA=(--cuda-devices 0,1,2)
elif [[ "${NUM_GPUS}" -eq 2 ]]; then
  POOL_EXTRA=(--cuda-devices 0,1)
fi
"${PYTHON}" "${REPO}/scripts/baseline_patient_pool_embeddings_dbt.py" \
  --checkpoint "${CKPT}" \
  --data-repo-root "${DATA_ROOT}" \
  --eval-splits val,test \
  "${POOL_EXTRA[@]}" \
  --batch-size 192 \
  --mode patient_attention \
  --artifacts-dir "${ATTN_DIR}" \
  --save-json "${ATTN_DIR}/metrics.json" \
  2>&1 | tee "${ATTN_DIR}/run.log"

PROBE_DIR="${RUN_DIR}/probe_linear_slice_head"
mkdir -p "${PROBE_DIR}"
"${PYTHON}" "${REPO}/scripts/train_linear_probe_dbt.py" \
  --checkpoint "${CKPT}" \
  --data-repo-root "${DATA_ROOT}" \
  --probe-out-dir "${PROBE_DIR}" \
  --device cuda \
  2>&1 | tee "${PROBE_DIR}/train.log"

echo "Done. Artifacts under: ${RUN_DIR}"
