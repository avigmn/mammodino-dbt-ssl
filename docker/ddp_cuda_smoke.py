#!/usr/bin/env python3
"""Minimal 2+ GPU DDP CUDA smoke test."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside container")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"Visible GPUs ({torch.cuda.device_count()}) < world_size ({world_size})"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    x = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2.0
    ok = bool(torch.allclose(x.cpu(), torch.tensor([expected])))

    print(
        f"[rank {rank}] local_rank={local_rank} "
        f"cuda_device={torch.cuda.current_device()} all_reduce={x.item():.1f} ok={ok}",
        flush=True,
    )
    if not ok:
        raise RuntimeError(
            f"DDP all_reduce mismatch on rank {rank}: got {x.item():.1f}, expected {expected:.1f}"
        )

    dist.barrier()
    if rank == 0:
        print("DDP CUDA smoke test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
