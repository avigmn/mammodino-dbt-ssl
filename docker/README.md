# Docker Multi-GPU (DDP) for mammodino_ssl_project

This setup keeps everything under `/mnt/md0/Liron/` and does not touch the host NVIDIA driver.

## One command to run all checks

From `/mnt/md0/Liron/mammodino_ssl_project`:

```bash
docker/run_docker_checks.sh
```

This script does:

1. Build image `mammodino-ddp:cu121` from `docker/Dockerfile.ddp`
2. Verify CUDA in container (`torch.cuda.is_available()` and GPU count)
3. Verify a 2-GPU NCCL/DDP all-reduce smoke
4. Run a tiny 2-GPU `scripts/train_dino.py` smoke with `configs/dino_dbt_docker_smoke.yaml`

## Keep single-GPU fallback unchanged

Your existing host workflow is unchanged. You can still run:

```bash
.venv/bin/python scripts/train_dino.py --config configs/dino_dbt.yaml --device cuda --num-gpus 1
```
