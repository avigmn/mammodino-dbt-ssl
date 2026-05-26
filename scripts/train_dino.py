#!/usr/bin/env python3
"""Train DINO (phase-1) on DBT manifest + patient split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DBT_SRC = _REPO_ROOT.parent / "dbt_simclr_project" / "src"
for p in (_REPO_ROOT / "src", _DBT_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from mammodino_ssl.data import DBTDINODataset, DINODataConfig
from mammodino_ssl.models.dino_ssl import create_dino_ssl
from mammodino_ssl.train import DINOLogitCenter, DINOTrainer, collate_dino_views


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_paths(repo_root: Path, data_cfg_relpath: str) -> tuple[Path, Path, dict]:
    cfg_path = repo_root / data_cfg_relpath
    data_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    artifacts_rel = data_cfg.get("artifacts_dir", "artifacts")
    artifacts = repo_root / artifacts_rel
    manifest_path = artifacts / data_cfg.get("manifest_rel_path", "manifests/master_manifest.parquet")
    split_path = artifacts / data_cfg.get("split_rel_path", "splits/patient_split_v1.json")
    return manifest_path, split_path, data_cfg


def _cache_dir(repo_root: Path, data_cfg: dict, use_cache: bool) -> Path | None:
    if not use_cache:
        return None
    rel = Path(data_cfg.get("processed_cache_rel_path", "processed/cache"))
    if rel.is_absolute():
        return rel
    return repo_root / rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINO (TinyViT) on DBT slices.")
    parser.add_argument("--config", default="configs/dino_dbt.yaml", help="Path to DINO DBT yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default=None, help="Override output run_name base from yaml")
    parser.add_argument("--epochs", type=int, default=0, help="Override train.epochs from config (0 keeps config value)")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=-1,
        help="Override train.early_stopping.patience (-1 = use yaml; 0 = disable early stopping)",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help=(
            "If set, all artifacts for this run go under <bundle-root>/<run_name>/{checkpoints,logs} "
            "(single folder tree per run). Default: experiments/dbt_dino_runs/{checkpoints,logs}/<run_name>."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of GPUs for DDP training (use torchrun when >1)",
    )
    parser.add_argument(
        "--multi-gpu-mode",
        default="ddp",
        choices=["ddp", "dp"],
        help="Multi-GPU mode when --num-gpus > 1: ddp (torchrun) or dp (single-process DataParallel)",
    )
    args = parser.parse_args()

    cfg_path = _REPO_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    ddp_enabled = args.num_gpus > 1 and args.multi_gpu_mode == "ddp"
    dp_enabled = args.num_gpus > 1 and args.multi_gpu_mode == "dp"
    rank = 0
    local_rank = 0
    world_size = 1
    if ddp_enabled:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        rank = int(os.environ.get("RANK", "-1"))
        world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        if local_rank < 0 or rank < 0 or world_size < 0:
            raise RuntimeError(
                "For multi-GPU run with --num-gpus > 1, launch via torchrun, e.g. "
                "`torchrun --standalone --nproc_per_node=2 scripts/train_dino.py ... --num-gpus 2`."
            )
        if world_size != args.num_gpus:
            raise RuntimeError(f"WORLD_SIZE={world_size} must match --num-gpus={args.num_gpus}")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    elif dp_enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for multi-GPU DataParallel mode")
        if torch.cuda.device_count() < args.num_gpus:
            raise RuntimeError(
                f"Requested --num-gpus={args.num_gpus} but only {torch.cuda.device_count()} are visible. "
                "Set CUDA_VISIBLE_DEVICES accordingly."
            )

    data_root = Path(cfg.get("data", {}).get("repo_root", "../dbt_simclr_project"))
    if not data_root.is_absolute():
        data_root = (_REPO_ROOT / data_root).resolve()
    data_cfg_relpath = str(cfg.get("data", {}).get("data_config_path", "configs/data.yaml"))
    manifest_path, split_path, data_cfg = _resolve_data_paths(data_root, data_cfg_relpath)
    model_yaml = cfg.get("model", {})

    ds_yaml = cfg.get("dataset", {})
    use_cache = bool(ds_yaml.get("use_processed_cache", data_cfg.get("use_processed_cache", False)))
    ds_cfg = DINODataConfig(
        resize_height=int(ds_yaml.get("resize_height", data_cfg.get("resize_height", 384))),
        resize_width=int(ds_yaml.get("resize_width", data_cfg.get("resize_width", 384))),
        normalize=bool(ds_yaml.get("normalize", data_cfg.get("normalize", True))),
        split_seed=seed,
        use_processed_cache=use_cache,
        processed_cache_dir=_cache_dir(data_root, data_cfg, use_cache),
        processed_cache_token=str(data_cfg.get("processed_cache_token", "")),
        max_bad_sample_retries=int(ds_yaml.get("max_bad_sample_retries", 32)),
        volume_pair_mode=str(ds_yaml.get("volume_pair_mode", "off")),
        teacher_rrc_scale_min=float(ds_yaml.get("teacher_rrc_scale_min", 0.85)),
        teacher_rrc_scale_max=float(ds_yaml.get("teacher_rrc_scale_max", 1.0)),
        teacher_horizontal_flip_p=float(ds_yaml.get("teacher_horizontal_flip_p", 0.5)),
        student_rrc_scale_min=float(ds_yaml.get("student_rrc_scale_min", 0.2)),
        student_rrc_scale_max=float(ds_yaml.get("student_rrc_scale_max", 1.0)),
        student_horizontal_flip_p=float(ds_yaml.get("student_horizontal_flip_p", 0.5)),
        student_color_jitter_p=float(ds_yaml.get("student_color_jitter_p", 0.8)),
        student_color_jitter_brightness=float(ds_yaml.get("student_color_jitter_brightness", 0.2)),
        student_color_jitter_contrast=float(ds_yaml.get("student_color_jitter_contrast", 0.2)),
        student_gaussian_blur_p=float(ds_yaml.get("student_gaussian_blur_p", 0.5)),
        student_gaussian_blur_kernel=int(ds_yaml.get("student_gaussian_blur_kernel", 23)),
        stochastic_val_views=bool(ds_yaml.get("stochastic_val_views", False)),
        tissue_percentile=float(ds_yaml.get("tissue_percentile", 80.0)),
        tissue_mask_cleanup=bool(ds_yaml.get("tissue_mask_cleanup", True)),
        tissue_crop_bias_prob=float(ds_yaml.get("tissue_crop_bias_prob", 0.6)),
        tissue_crop_pad_frac=float(ds_yaml.get("tissue_crop_pad_frac", 0.08)),
        patch_size=int(model_yaml.get("patch_size", 16)),
    )

    train_ds = DBTDINODataset(
        manifest_path=manifest_path,
        split_path=split_path,
        split="train",
        config=ds_cfg,
    )
    val_ds = DBTDINODataset(
        manifest_path=manifest_path,
        split_path=split_path,
        split="val",
        config=ds_cfg,
    )
    train_ds.set_epoch(0)

    train_yaml = cfg.get("train", {})
    batch_size = int(train_yaml.get("batch_size", 8))
    num_workers = int(train_yaml.get("num_workers", 4))
    pin_memory = bool(train_yaml.get("pin_memory", True))
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp_enabled else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp_enabled else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dino_views,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dino_views,
        drop_last=False,
    )

    model = create_dino_ssl(
        image_size=int(model_yaml.get("image_size", 384)),
        num_prototypes=int(model_yaml.get("num_prototypes", 512)),
        num_patch_prototypes=int(model_yaml.get("num_patch_prototypes", model_yaml.get("num_prototypes", 512))),
        embed_dim=int(model_yaml.get("embed_dim", 192)),
        depth=int(model_yaml.get("depth", 4)),
        num_heads=int(model_yaml.get("num_heads", 3)),
        head_hidden_dim=int(model_yaml.get("head_hidden_dim", 512)),
        head_bottleneck_dim=int(model_yaml.get("head_bottleneck_dim", 256)),
    )
    center = DINOLogitCenter(
        int(model_yaml.get("num_prototypes", 512)),
        center_momentum=float(train_yaml.get("center_momentum", 0.9)),
    )
    patch_center = DINOLogitCenter(
        int(model_yaml.get("num_patch_prototypes", model_yaml.get("num_prototypes", 512))),
        center_momentum=float(train_yaml.get("ibot_center_momentum", train_yaml.get("center_momentum", 0.9))),
    )

    output_yaml = cfg.get("output", {})
    base_run_name = args.run_name or str(output_yaml.get("run_name", "dino_dbt_run"))
    auto_timestamp = bool(output_yaml.get("auto_append_timestamp", True))
    run_name = f"{base_run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}" if auto_timestamp else base_run_name
    if args.bundle_root is not None:
        bundle_parent = args.bundle_root if args.bundle_root.is_absolute() else (_REPO_ROOT / args.bundle_root).resolve()
        run_dir = bundle_parent / run_name
        ckpt_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
    else:
        artifacts_root = _REPO_ROOT / "experiments" / "dbt_dino_runs"
        ckpt_dir = artifacts_root / "checkpoints" / run_name
        logs_dir = artifacts_root / "logs" / run_name

    device_name = args.device
    if ddp_enabled:
        if not device_name.startswith("cuda"):
            raise RuntimeError("DDP multi-GPU requires --device cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DDP multi-GPU mode")
        device_name = f"cuda:{local_rank}"
    elif dp_enabled:
        if not device_name.startswith("cuda"):
            raise RuntimeError("DataParallel multi-GPU requires --device cuda")
        device_name = "cuda:0"
    elif device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    trainer = DINOTrainer(
        model=model,
        center=center,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=float(train_yaml.get("learning_rate", 1e-4)),
        weight_decay=float(train_yaml.get("weight_decay", 1e-4)),
        teacher_momentum=float(train_yaml.get("teacher_momentum", 0.996)),
        student_temp=float(train_yaml.get("student_temp", 0.1)),
        teacher_temp=float(train_yaml.get("teacher_temp", 0.04)),
        patch_center=patch_center,
        ibot_weight=float(train_yaml.get("w_ibot", 0.0)),
        ibot_mask_ratio=float(train_yaml.get("ibot_mask_ratio", 0.35)),
        ibot_student_temp=float(train_yaml.get("ibot_student_temp", train_yaml.get("student_temp", 0.1))),
        ibot_teacher_temp=float(train_yaml.get("ibot_teacher_temp", train_yaml.get("teacher_temp", 0.04))),
        grad_accum_steps=int(train_yaml.get("grad_accum_steps", 1)),
        amp=bool(train_yaml.get("amp", True)),
        checkpoints_dir=ckpt_dir,
        logs_dir=logs_dir,
        ddp=ddp_enabled,
        ddp_rank=rank,
        data_parallel=dp_enabled,
        data_parallel_device_ids=list(range(args.num_gpus)) if dp_enabled else None,
    )

    early = train_yaml.get("early_stopping") or {}
    if int(args.early_stopping_patience) >= 0:
        es_patience = int(args.early_stopping_patience)
    else:
        es_patience = int(early.get("patience", 0) or 0)
    cfg_epochs = int(train_yaml.get("epochs", 30))

    effective_epochs = args.epochs if args.epochs > 0 else cfg_epochs
    if rank == 0 and args.bundle_root is not None:
        run_dir_for_meta = ckpt_dir.parent
        run_dir_for_meta.mkdir(parents=True, exist_ok=True)
        snap = {
            "run_name": run_name,
            "epochs_requested": effective_epochs,
            "early_stopping_patience": es_patience,
            "early_stopping_min_delta": float(early.get("min_delta", 0.0)),
            "run_bundle_dir": str(run_dir_for_meta.resolve()),
            "checkpoints_dir": str(ckpt_dir.resolve()),
            "logs_dir": str(logs_dir.resolve()),
            "config_path": str(cfg_path.resolve()),
            "note": "CLI epochs / early-stopping patience mirrored in effective_train_config.yaml.",
        }
        (run_dir_for_meta / "run_manifest.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
        merged_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        merged_cfg.setdefault("train", {})
        merged_cfg["train"]["epochs"] = effective_epochs
        merged_cfg["train"].setdefault("early_stopping", {})
        merged_cfg["train"]["early_stopping"]["patience"] = es_patience
        merged_cfg["train"]["early_stopping"]["min_delta"] = float(early.get("min_delta", 0.0))
        (run_dir_for_meta / "effective_train_config.yaml").write_text(yaml.safe_dump(merged_cfg, sort_keys=False), encoding="utf-8")
    out = trainer.fit(
        epochs=effective_epochs,
        max_train_steps=int(train_yaml.get("max_train_steps", 0)) or None,
        max_val_steps=int(train_yaml.get("max_val_steps", 0)) or None,
        early_stopping_patience=(es_patience if es_patience > 0 else None),
        early_stopping_min_delta=float(early.get("min_delta", 0.0)),
    )

    if rank == 0:
        print("DINO DBT training complete")
        print(f"run_name={run_name}")
        print(f"best_val_loss={out.best_val_loss:.4f} best_epoch={out.best_epoch}")
        print(f"best_checkpoint={out.best_checkpoint}")
        print(f"history={out.history_path}")
        print(f"summary={out.summary_path}")
        print(f"plots_dir={out.plots_dir}")

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
