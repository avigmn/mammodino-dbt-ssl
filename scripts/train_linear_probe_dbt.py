#!/usr/bin/env python3
"""Linear probe for DBT after DINO pretraining."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DBT_SRC = _REPO_ROOT.parent / "dbt_simclr_project" / "src"
for p in (_REPO_ROOT / "src", _DBT_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from dbt_ssl.data.supervised_slice_dataset import DBTSupervisedSliceDataset, SupervisedSliceConfig
from mammodino_ssl.models.dino_ssl import create_dino_ssl
from mammodino_ssl.models.linear_probe import FrozenTinyViTLinearProbe


def _normalize_ddp_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert DDP-wrapped submodule keys to plain module keys."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        nk = nk.replace("student_backbone.module.", "student_backbone.")
        nk = nk.replace("student_head.module.", "student_head.")
        nk = nk.replace("student_patch_head.module.", "student_patch_head.")
        out[nk] = v
    return out


def _load_dino_model_section(config_path: Path) -> dict:
    """YAML `model:` block from the same file used for SSL (spatial size + TinyViT dims)."""
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return dict(raw.get("model") or {})


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor3(image_nchw: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(image_nchw).float()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t


def collate_supervised(batch: list[dict]) -> dict[str, torch.Tensor]:
    x = torch.stack([_to_tensor3(item["image"]) for item in batch], dim=0)
    y = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    meta = [
        {
            "canonical_patient_id": item.get("canonical_patient_id"),
            "slice_abspath": item.get("slice_abspath"),
        }
        for item in batch
    ]
    return {"image": x, "label": y, "meta": meta}


@torch.no_grad()
def _eval_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_steps: int | None,
    loss_fn: nn.Module,
) -> tuple[float, int, int]:
    model.eval()
    amp_on = amp and device.type == "cuda"
    total_loss = 0.0
    correct = 0
    total = 0
    for step_idx, batch in enumerate(loader):
        if max_steps is not None and step_idx >= max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    return total_loss, correct, total


@torch.no_grad()
def _eval_roc_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    max_steps: int | None,
    positive_class: int,
) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    model.eval()
    amp_on = amp and device.type == "cuda"
    pc = int(positive_class)
    y_true: list[int] = []
    y_score: list[float] = []
    for step_idx, batch in enumerate(loader):
        if max_steps is not None and step_idx >= max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            prob = F.softmax(logits.float(), dim=1)[:, pc]
        y_true.extend(y.detach().cpu().tolist())
        y_score.extend(prob.detach().cpu().tolist())
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(np.asarray(y_true, dtype=np.int64), np.asarray(y_score, dtype=np.float64)))


def _ddp_sum(value: float, device: torch.device) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _label_distribution(df: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(df))
    n0 = int((df["label_clinical"] == 0).sum())
    n1 = int((df["label_clinical"] == 1).sum())
    denom = max(1, total)
    return {
        "n_samples": total,
        "n_class_0": n0,
        "n_class_1": n1,
        "p_class_0": float(n0 / denom),
        "p_class_1": float(n1 / denom),
    }


def _compute_inverse_freq_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    n0 = int((train_df["label_clinical"] == 0).sum())
    n1 = int((train_df["label_clinical"] == 1).sum())
    if n0 <= 0 or n1 <= 0:
        raise ValueError(f"inverse_freq requires both classes in train split, got n0={n0}, n1={n1}")
    n = n0 + n1
    w0 = float(n / (2.0 * n0))
    w1 = float(n / (2.0 * n1))
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def _make_subset(ds: DBTSupervisedSliceDataset, n: int, *, seed: int, split_name: str) -> Subset:
    if n <= 0:
        raise ValueError(f"{split_name} subset size must be > 0, got {n}")
    if len(ds) == 0:
        raise ValueError(f"{split_name} dataset is empty")
    n = min(int(n), len(ds))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(ds), size=n, replace=False).tolist()
    return Subset(ds, idx)


@torch.no_grad()
def _log_feature_stats(
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    max_steps: int = 1,
) -> None:
    """Print basic statistics for frozen CLS embeddings and head logits."""
    probe.eval()
    amp_on = amp and device.type == "cuda"
    steps = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        meta = batch.get("meta")
        with torch.amp.autocast("cuda", enabled=amp_on):
            # Access encoder+head if wrapped by DDP.
            m = probe.module if hasattr(probe, "module") else probe  # type: ignore[assignment]
            enc = getattr(m, "encoder", None)
            head = getattr(m, "head", None)
            if enc is None or head is None:
                print("[debug] feature_stats: probe missing encoder/head attributes", flush=True)
                return
            cls, _ = enc(x)
            logits = head(cls)
        cls_f = cls.float()
        logits_f = logits.float()
        cls_n = torch.nn.functional.normalize(cls_f, dim=1)
        sim = cls_n @ cls_n.t()
        sim.fill_diagonal_(-1.0)
        max_cos = float(sim.max().item()) if sim.numel() else float("nan")
        if sim.numel():
            flat_idx = int(sim.argmax().item())
            i = flat_idx // sim.shape[1]
            j = flat_idx % sim.shape[1]
            yi = int(y[i].item())
            yj = int(y[j].item())
            meta_str = ""
            if isinstance(meta, list) and i < len(meta) and j < len(meta):
                mi = meta[i] or {}
                mj = meta[j] or {}
                meta_str = (
                    f" pair_i(label={yi},patient={mi.get('canonical_patient_id')},path={mi.get('slice_abspath')})"
                    f" pair_j(label={yj},patient={mj.get('canonical_patient_id')},path={mj.get('slice_abspath')})"
                )
            else:
                meta_str = f" pair_i_label={yi} pair_j_label={yj}"
        else:
            meta_str = ""
        nan_frac = float(torch.isnan(cls_f).float().mean().item())
        inf_frac = float(torch.isinf(cls_f).float().mean().item())
        print(
            "[debug] feature_stats "
            f"cls_shape={tuple(cls_f.shape)} "
            f"cls_mean={cls_f.mean().item():.4g} cls_std={cls_f.std(unbiased=False).item():.4g} "
            f"cls_min={cls_f.min().item():.4g} cls_max={cls_f.max().item():.4g} "
            f"nan_frac={nan_frac:.3g} inf_frac={inf_frac:.3g} "
            f"max_cos_sim={max_cos:.4g}{meta_str} "
            f"logits_mean={logits_f.mean().item():.4g} logits_std={logits_f.std(unbiased=False).item():.4g} "
            f"y_pos_frac={(y == 1).float().mean().item():.3g}",
            flush=True,
        )
        steps += 1
        if steps >= max_steps:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe on DBT from DINO checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to DINO best checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--dino-config",
        type=Path,
        default=_REPO_ROOT / "configs/dino_dbt.yaml",
        help="SSL training YAML; `model.image_size` (224 for dino_dbt) and architecture must match the checkpoint.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override model.image_size from --dino-config (default: yaml, else 224).",
    )
    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=None,
        help="Override model.num_prototypes from --dino-config (default: yaml, else 512).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of GPUs for DDP training (launch with torchrun when >1).",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max-val-steps", type=int, default=0, help="0 means full validation epoch")
    parser.add_argument(
        "--data-repo-root",
        type=Path,
        default=Path("../dbt_simclr_project"),
        help="Repo containing artifacts/manifests and data config",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Do not write confusion matrix / ROC / eval_metrics after training (default: run full-val eval).",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Do not load test split or run test-side confusion / patient-level metrics after training.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=0,
        help="Cap batches for final confusion+ROC pass (0 = full validation loader).",
    )
    parser.add_argument("--positive-class", type=int, default=1, help="Class index for ROC probability column.")
    parser.add_argument(
        "--pos-class-weight",
        type=float,
        default=2.0,
        help="CrossEntropy weight on the positive class (true label = positive); >1 penalizes missing a positive more. Set 1.0 for uniform classes.",
    )
    parser.add_argument(
        "--ce-weight-mode",
        choices=["asymmetric", "none", "inverse_freq"],
        default="asymmetric",
        help=(
            "How CrossEntropy weights are set: asymmetric uses --pos-class-weight on positive class, "
            "none uses unweighted CE, inverse_freq uses train-split inverse-frequency weights."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop if val metric does not improve for this many epochs. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum val_acc gain (monitor=val_acc) or val_loss drop (monitor=val_loss) to count as improvement.",
    )
    parser.add_argument(
        "--early-stopping-monitor",
        choices=["val_acc", "val_loss", "val_roc_auc"],
        default="val_acc",
        help="Metric for best checkpoint and early stopping.",
    )
    parser.add_argument(
        "--probe-head",
        choices=["linear", "mlp"],
        default="linear",
        help="Probe head type on top of frozen CLS features.",
    )
    parser.add_argument(
        "--probe-mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden dim for MLP head (when --probe-head mlp). Default: max(128, embed_dim).",
    )
    parser.add_argument(
        "--probe-dropout",
        type=float,
        default=0.0,
        help="Dropout for MLP head (when --probe-head mlp).",
    )
    parser.add_argument(
        "--overfit-train-samples",
        type=int,
        default=0,
        help="Sanity check: train on a small random subset (0 disables).",
    )
    parser.add_argument(
        "--overfit-use-train-as-val",
        action="store_true",
        help="Sanity check: evaluate on the same subset as train (useful to verify convergence quickly).",
    )
    parser.add_argument(
        "--debug-feature-stats-steps",
        type=int,
        default=0,
        help="If >0, print frozen CLS embedding/logit stats for this many loader batches at start of training.",
    )
    parser.add_argument(
        "--debug-print-samples",
        type=int,
        default=0,
        help="If >0 (main process only), print this many (path,label,patient) samples from train/val datasets.",
    )
    parser.add_argument(
        "--debug-head-grad-steps",
        type=int,
        default=0,
        help="If >0 (main process only), print head param/grad norms for the first N optimizer steps of epoch 1.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default="",
        help=(
            "Optional subfolder under experiments/dbt_dino_runs/probe/ to group related runs "
            "(example: sanity_checks_may09)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Optional descriptive run directory name (example: overfit512_mlp_lr1e3_3gpu). "
            "If empty, defaults to probe_<timestamp>."
        ),
    )
    parser.add_argument(
        "--probe-out-dir",
        type=Path,
        default=None,
        help=(
            "If set, write probe artifacts (probe_best.pt, confusion_eval/, probe_summary.json) "
            "into this directory directly instead of experiments/dbt_dino_runs/probe/...."
        ),
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    ddp_enabled = args.num_gpus > 1
    rank = 0
    local_rank = 0
    world_size = 1
    if ddp_enabled:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        rank = int(os.environ.get("RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        if local_rank < 0 or rank < 0 or world_size < 0:
            raise RuntimeError(
                "For --num-gpus > 1, run with torchrun, e.g. "
                "`torchrun --standalone --nproc_per_node=3 scripts/train_linear_probe_dbt.py ... --num-gpus 3`."
            )
        if world_size != args.num_gpus:
            raise RuntimeError(f"WORLD_SIZE={world_size} must match --num-gpus={args.num_gpus}")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    main_rank = rank == 0
    dino_cfg_path = args.dino_config if args.dino_config.is_absolute() else (_REPO_ROOT / args.dino_config).resolve()
    model_yaml = _load_dino_model_section(dino_cfg_path)
    image_size = int(args.image_size if args.image_size is not None else model_yaml.get("image_size", 224))
    num_prototypes = int(args.num_prototypes if args.num_prototypes is not None else model_yaml.get("num_prototypes", 512))

    data_root = args.data_repo_root if args.data_repo_root.is_absolute() else (_REPO_ROOT / args.data_repo_root).resolve()
    data_cfg_path = data_root / "configs" / "data.yaml"

    data_cfg = yaml.safe_load(data_cfg_path.read_text(encoding="utf-8"))
    artifacts = data_root / data_cfg.get("artifacts_dir", "artifacts")
    manifest_path = artifacts / data_cfg.get("manifest_rel_path", "manifests/master_manifest.parquet")
    split_path = artifacts / data_cfg.get("split_rel_path", "splits/patient_split_v1.json")
    use_cache = bool(data_cfg.get("use_processed_cache", False))
    cache_dir = Path(data_cfg.get("processed_cache_rel_path", "")) if use_cache else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = data_root / cache_dir
    manifest_df = pd.read_parquet(manifest_path)
    split_map = json.loads(split_path.read_text(encoding="utf-8"))
    manifest_df = manifest_df.copy()
    manifest_df["split"] = manifest_df["canonical_patient_id"].map(split_map)
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    train_distribution = _label_distribution(train_df)
    val_distribution = _label_distribution(val_df)
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    test_distribution = _label_distribution(test_df) if len(test_df) else None
    majority_class = 0 if int(val_distribution["n_class_0"]) >= int(val_distribution["n_class_1"]) else 1
    majority_baseline = float(max(float(val_distribution["p_class_0"]), float(val_distribution["p_class_1"])))

    ds_cfg = SupervisedSliceConfig(
        resize_height=image_size,
        resize_width=image_size,
        normalize=bool(data_cfg.get("normalize", True)),
        split_seed=args.seed,
        use_processed_cache=use_cache,
        processed_cache_dir=cache_dir,
        processed_cache_token=str(data_cfg.get("processed_cache_token", "")),
    )
    train_ds = DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path, split="train", config=ds_cfg)
    val_ds = DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path, split="val", config=ds_cfg)
    test_ds = None
    if main_rank and int(args.overfit_train_samples) <= 0 and not args.skip_test_eval:
        test_ds = DBTSupervisedSliceDataset(
            manifest_path=manifest_path, split_path=split_path, split="test", config=ds_cfg
        )
    if int(args.overfit_train_samples) > 0:
        train_ds = _make_subset(train_ds, int(args.overfit_train_samples), seed=args.seed, split_name="train")  # type: ignore[assignment]
        if args.overfit_use_train_as_val:
            val_ds = train_ds  # type: ignore[assignment]
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp_enabled else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp_enabled else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_supervised,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_supervised,
    )
    # For AUROC monitoring on capped validation steps, sample val randomly to avoid single-class windows.
    val_auc_loader: DataLoader | None = None
    if not ddp_enabled and args.early_stopping_monitor == "val_roc_auc":
        g = torch.Generator()
        g.manual_seed(int(args.seed))
        val_auc_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=collate_supervised,
            generator=g,
        )

    device_name = args.device
    if ddp_enabled:
        if not device_name.startswith("cuda"):
            raise RuntimeError("DDP multi-GPU probe requires --device cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DDP probe mode")
        device_name = f"cuda:{local_rank}"
    elif device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    amp = (not args.no_amp) and device.type == "cuda"

    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (_REPO_ROOT / args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {ckpt_path}")

    ssl = create_dino_ssl(
        image_size=image_size,
        num_prototypes=num_prototypes,
        embed_dim=int(model_yaml.get("embed_dim", 192)),
        depth=int(model_yaml.get("depth", 4)),
        num_heads=int(model_yaml.get("num_heads", 3)),
        head_hidden_dim=int(model_yaml.get("head_hidden_dim", 512)),
        head_bottleneck_dim=int(model_yaml.get("head_bottleneck_dim", 256)),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt["model_state"]
    try:
        ssl.load_state_dict(model_state, strict=True)
    except RuntimeError:
        # Allow probe to load checkpoints that were saved while DDP wrapped submodules.
        ssl.load_state_dict(_normalize_ddp_state_dict_keys(model_state), strict=True)
    probe = FrozenTinyViTLinearProbe(
        ssl.student_backbone,
        num_classes=2,
        head_type=args.probe_head,
        mlp_hidden_dim=args.probe_mlp_hidden_dim,
        dropout=args.probe_dropout,
    ).to(device)
    if ddp_enabled:
        probe = DDP(
            probe,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    pos_c = int(args.positive_class)
    if pos_c not in (0, 1):
        raise ValueError("--positive-class must be 0 or 1 for binary probe")
    ce_weight: torch.Tensor | None = None
    if args.ce_weight_mode == "none":
        loss_fn = nn.CrossEntropyLoss()
    elif args.ce_weight_mode == "inverse_freq":
        ce_weight = _compute_inverse_freq_weights(train_df, device)
        loss_fn = nn.CrossEntropyLoss(weight=ce_weight)
    else:
        ce_weight = torch.ones(2, device=device, dtype=torch.float32)
        ce_weight[pos_c] = float(args.pos_class_weight)
        loss_fn = nn.CrossEntropyLoss(weight=ce_weight)

    probe_head = probe.module.head if ddp_enabled else probe.head  # type: ignore[attr-defined]
    opt = AdamW(probe_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_roc_auc = float("nan")
    best_epoch = -1
    es_patience = int(args.early_stopping_patience)
    es_min_delta = float(args.early_stopping_min_delta)
    es_monitor = args.early_stopping_monitor
    epochs_no_improve = 0
    stopped_early = False
    is_main_process = main_rank

    if is_main_process and int(args.debug_print_samples) > 0:
        n = int(args.debug_print_samples)
        print(f"[debug] printing {n} samples from train/val datasets", flush=True)
        for split_name, ds in (("train", train_ds), ("val", val_ds)):
            try:
                m = min(n, len(ds))  # type: ignore[arg-type]
            except TypeError:
                # Some datasets (Subset) are fine; this is just defensive.
                m = n
            for i in range(m):
                item = ds[i]  # type: ignore[index]
                print(
                    f"[debug] {split_name}[{i}] "
                    f"label={item.get('label')} "
                    f"patient={item.get('canonical_patient_id')} "
                    f"path={item.get('slice_abspath')}",
                    flush=True,
                )

    if is_main_process and int(args.debug_feature_stats_steps) > 0:
        _log_feature_stats(
            probe,
            train_loader,
            device,
            amp=amp,
            max_steps=int(args.debug_feature_stats_steps),
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_group = str(args.run_group).strip()
    run_name = str(args.run_name).strip()
    if args.probe_out_dir is not None:
        out_dir = args.probe_out_dir if args.probe_out_dir.is_absolute() else (_REPO_ROOT / args.probe_out_dir).resolve()
    else:
        probe_root = _REPO_ROOT / "experiments" / "dbt_dino_runs" / "probe"
        if run_group:
            probe_root = probe_root / run_group
        if run_name:
            safe_name = run_name.replace(" ", "_")
            out_dir = probe_root / f"{safe_name}_{stamp}"
        else:
            out_dir = probe_root / f"probe_{stamp}"
    if is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp_enabled:
        dist.barrier()
    run_config = {
        "checkpoint": str(ckpt_path),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "device_requested": args.device,
        "device_actual": str(device),
        "ce_weight_mode": args.ce_weight_mode,
        "positive_class": pos_c,
        "pos_class_weight": float(args.pos_class_weight),
        "ce_weight_vector": ce_weight.detach().cpu().tolist() if ce_weight is not None else None,
        "train_label_distribution": train_distribution,
        "val_label_distribution": val_distribution,
        "majority_class_val": majority_class,
        "majority_baseline_val_accuracy": majority_baseline,
        "max_train_steps": int(args.max_train_steps),
        "max_val_steps": int(args.max_val_steps),
        "eval_max_steps": int(args.eval_max_steps),
        "run_group": run_group if run_group else None,
        "run_name": run_name if run_name else None,
    }
    if is_main_process:
        (out_dir / "probe_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        probe.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_n = 0
        pbar = tqdm(
            train_loader,
            desc=f"probe {epoch+1}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
            disable=not is_main_process,
        )
        max_train_steps = args.max_train_steps if args.max_train_steps > 0 else None
        for step_idx, batch in enumerate(pbar):
            if max_train_steps is not None and step_idx >= max_train_steps:
                break
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = probe(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if is_main_process and epoch == 0 and int(args.debug_head_grad_steps) > 0 and step_idx < int(args.debug_head_grad_steps):
                head_module = probe.module.head if ddp_enabled else probe.head  # type: ignore[attr-defined]
                with torch.no_grad():
                    pnorm = 0.0
                    gnorm = 0.0
                    n_p = 0
                    n_g = 0
                    for p in head_module.parameters():
                        pnorm += float(p.detach().float().norm().item())
                        n_p += 1
                        if p.grad is not None:
                            gnorm += float(p.grad.detach().float().norm().item())
                            n_g += 1
                    print(
                        f"[debug] head_norms step={step_idx} "
                        f"param_norm_sum={pnorm:.4g} (n={n_p}) grad_norm_sum={gnorm:.4g} (n={n_g}) "
                        f"loss={float(loss.item()):.4g}",
                        flush=True,
                    )
            scaler.step(opt)
            scaler.update()
            train_loss_sum += float(loss.item()) * y.size(0)
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_n += int(y.numel())
        if ddp_enabled:
            train_loss_sum = _ddp_sum(train_loss_sum, device)
            train_correct = int(_ddp_sum(float(train_correct), device))
            train_n = int(_ddp_sum(float(train_n), device))
        train_loss = train_loss_sum / max(1, train_n)
        train_acc = train_correct / max(1, train_n)
        max_val_steps = args.max_val_steps if args.max_val_steps > 0 else None
        val_loss_sum_local, val_correct_local, val_n_local = _eval_stats(
            probe, val_loader, device, amp=amp, max_steps=max_val_steps, loss_fn=loss_fn
        )
        auc_loader = val_auc_loader if val_auc_loader is not None else val_loader
        val_roc_auc_local = _eval_roc_auc(
            probe,
            auc_loader,
            device,
            amp=amp,
            max_steps=max_val_steps,
            positive_class=pos_c,
        )
        if ddp_enabled:
            val_loss_sum = _ddp_sum(val_loss_sum_local, device)
            val_correct_sum = _ddp_sum(float(val_correct_local), device)
            val_n_sum = _ddp_sum(float(val_n_local), device)
            # AUROC isn't reducible; compute on rank0 only (DDP probe is for speed, not metric exactness).
            val_roc_auc = val_roc_auc_local if is_main_process else float("nan")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": (val_loss_sum / max(1.0, val_n_sum)) if ddp_enabled else (val_loss_sum_local / max(1, val_n_local)),
                "val_acc": (val_correct_sum / max(1.0, val_n_sum)) if ddp_enabled else (val_correct_local / max(1, val_n_local)),
                "val_roc_auc": val_roc_auc_local if not ddp_enabled else val_roc_auc,
            }
        )
        val_loss = history[-1]["val_loss"]
        val_acc = history[-1]["val_acc"]
        val_roc_auc = history[-1]["val_roc_auc"]
        if is_main_process:
            print(
                f"probe epoch={epoch+1} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_roc_auc={val_roc_auc:.4f}",
                flush=True,
            )

        improved = False
        if es_monitor == "val_acc":
            if val_acc > best_val_acc + es_min_delta:
                improved = True
        else:
            if es_monitor == "val_loss":
                if val_loss < best_val_loss - es_min_delta:
                    improved = True
            else:
                if not np.isnan(val_roc_auc) and (np.isnan(best_val_roc_auc) or val_roc_auc > best_val_roc_auc + es_min_delta):
                    improved = True

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_roc_auc = val_roc_auc
            best_epoch = epoch + 1
            if is_main_process:
                head_module = probe.module.head if ddp_enabled else probe.head  # type: ignore[attr-defined]
                torch.save(
                    {"head_state": head_module.state_dict(), "val_acc": val_acc, "val_loss": val_loss, "val_roc_auc": val_roc_auc},
                    out_dir / "probe_best.pt",
                )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if es_patience > 0 and epochs_no_improve >= es_patience:
            stopped_early = True
            if is_main_process:
                print(
                    f"early stopping: no improvement on {es_monitor} for {es_patience} epochs "
                    f"(best_epoch={best_epoch})",
                    flush=True,
                )
            break

    summary = {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "best_val_roc_auc": None if np.isnan(best_val_roc_auc) else float(best_val_roc_auc),
        "best_epoch": best_epoch,
        "random_baseline_acc": 0.5,
        "majority_baseline_val_accuracy": majority_baseline,
        "majority_class_val": majority_class,
        "pos_class_weight": float(args.pos_class_weight),
        "ce_weight_mode": args.ce_weight_mode,
        "ce_weight_vector": ce_weight.detach().cpu().tolist() if ce_weight is not None else None,
        "positive_class": pos_c,
        "train_label_distribution": train_distribution,
        "val_label_distribution": val_distribution,
        "test_label_distribution": test_distribution,
        "early_stopping": {
            "patience": es_patience,
            "min_delta": es_min_delta,
            "monitor": es_monitor,
            "triggered": stopped_early,
            "stopped_epoch": epoch + 1 if stopped_early else None,
        },
    }
    if is_main_process:
        (out_dir / "probe_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_pt = out_dir / "probe_best.pt"
    eval_dir = out_dir / "confusion_eval"
    if is_main_process and not args.skip_final_eval and best_pt.is_file():
        try:
            # Lazy import so training can run without matplotlib/sklearn installed.
            from mammodino_ssl.eval import save_confusion_roc_metrics
            from mammodino_ssl.eval.dbt_probe_metrics import summarize_patient_level_mean_probs_val_and_test

            probe_sd = torch.load(best_pt, map_location=device, weights_only=False)
            if ddp_enabled:
                probe.module.head.load_state_dict(probe_sd["head_state"], strict=True)  # type: ignore[attr-defined]
            else:
                probe.head.load_state_dict(probe_sd["head_state"], strict=True)
            eval_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.device.startswith("cuda"),
                collate_fn=collate_supervised,
            )
            max_eval_steps = args.eval_max_steps if args.eval_max_steps > 0 else None
            eval_summary = save_confusion_roc_metrics(
                probe,
                eval_loader,
                device=device,
                amp=amp,
                out_dir=eval_dir,
                split="val",
                positive_class=args.positive_class,
                max_steps=max_eval_steps,
                dino_checkpoint=str(ckpt_path),
                probe_checkpoint=str(best_pt.resolve()),
                extra_fields={
                    "train_label_distribution": train_distribution,
                    "val_label_distribution": val_distribution,
                    "majority_class_val": majority_class,
                    "majority_baseline_val_accuracy": majority_baseline,
                    "ce_weight_mode": args.ce_weight_mode,
                    "ce_weight_vector": ce_weight.detach().cpu().tolist() if ce_weight is not None else None,
                },
            )
            paths = eval_summary.pop("_paths", {})
            summary["final_eval"] = {**paths, **{k: v for k, v in eval_summary.items() if k not in ("fpr", "tpr")}}

            test_loader_eval = None
            if test_ds is not None and len(test_ds) > 0:
                test_loader_eval = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.device.startswith("cuda"),
                    collate_fn=collate_supervised,
                )
                test_summary = save_confusion_roc_metrics(
                    probe,
                    test_loader_eval,
                    device=device,
                    amp=amp,
                    out_dir=eval_dir,
                    split="test",
                    positive_class=args.positive_class,
                    max_steps=max_eval_steps,
                    dino_checkpoint=str(ckpt_path),
                    probe_checkpoint=str(best_pt.resolve()),
                    extra_fields={
                        "train_label_distribution": train_distribution,
                        "test_label_distribution": test_distribution,
                        "ce_weight_mode": args.ce_weight_mode,
                        "ce_weight_vector": ce_weight.detach().cpu().tolist() if ce_weight is not None else None,
                    },
                )
                test_paths = test_summary.pop("_paths", {})
                summary["final_eval_test"] = {**test_paths, **{k: v for k, v in test_summary.items() if k not in ("fpr", "tpr")}}

            summary["patient_level_eval"] = summarize_patient_level_mean_probs_val_and_test(
                probe,
                eval_loader,
                test_loader_eval,
                device=device,
                amp=amp,
                positive_class=args.positive_class,
                max_steps_val=max_eval_steps,
                max_steps_test=max_eval_steps,
            )
            (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except ImportError as exc:
            print(f"final eval skipped (install scikit-learn): {exc}", flush=True)
    elif is_main_process and not args.skip_final_eval and not best_pt.is_file():
        print(f"final eval skipped (no {best_pt.name} written)", flush=True)

    if is_main_process:
        print(json.dumps({"probe_out_dir": str(out_dir), **summary}, indent=2))

    if ddp_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
