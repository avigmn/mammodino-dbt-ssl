#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/md0/Liron"
PROJ="${ROOT}/mammodino_ssl_project"
IMAGE="mammodino-ddp:cu121"

echo "[1/4] Building Docker image: ${IMAGE}"
docker build -f "${PROJ}/docker/Dockerfile.ddp" -t "${IMAGE}" "${PROJ}"

echo "[2/4] Verifying CUDA visibility in container"
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${ROOT}:${ROOT}" \
  -w "${PROJ}" \
  "${IMAGE}" \
  bash -lc 'python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"'

echo "[3/4] Verifying 2-GPU DDP all_reduce smoke"
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_NET=Socket \
  -e NCCL_SOCKET_IFNAME=^lo,docker0 \
  -e NCCL_NVLS_ENABLE=0 \
  -v "${ROOT}:${ROOT}" \
  -w "${PROJ}" \
  "${IMAGE}" \
  bash -lc 'torchrun --standalone --nnodes=1 --nproc_per_node=2 docker/ddp_cuda_smoke.py'

echo "[4/4] Running tiny MammoDINO 2-GPU smoke"
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_NET=Socket \
  -e NCCL_SOCKET_IFNAME=^lo,docker0 \
  -e NCCL_NVLS_ENABLE=0 \
  -v "${ROOT}:${ROOT}" \
  -w "${PROJ}" \
  "${IMAGE}" \
  bash -lc 'torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_dino.py --config configs/dino_dbt_docker_smoke.yaml --device cuda --num-gpus 2 --multi-gpu-mode ddp --run-name docker_ddp_smoke'

echo "Done. Docker CUDA + DDP + project smoke all passed."
