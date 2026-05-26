#!/usr/bin/env python3
"""Baseline: fit sklearn LogisticRegression on frozen DINO CLS embeddings (DBT)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DBT_SRC = _REPO_ROOT.parent / "dbt_simclr_project" / "src"
for p in (_REPO_ROOT / "src", _DBT_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from dbt_ssl.data.supervised_slice_dataset import DBTSupervisedSliceDataset, SupervisedSliceConfig
from mammodino_ssl.models.dino_ssl import create_dino_ssl


def _load_dino_model_section(config_path: Path) -> dict:
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return dict(raw.get("model") or {})


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


def _to_tensor3(image_nchw: object) -> torch.Tensor:
    t = torch.from_numpy(image_nchw).float()  # type: ignore[arg-type]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t


def collate_supervised(batch: list[dict]) -> dict[str, object]:
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
def _extract_cls_embeddings(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    amp_on = amp and device.type == "cuda"
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for step_idx, batch in enumerate(loader):
        if max_batches is not None and step_idx >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            cls, _ = backbone(x)
        xs.append(cls.detach().float().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
    Y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser(description="LogReg baseline on frozen DINO CLS embeddings (DBT).")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to DINO best checkpoint")
    parser.add_argument(
        "--dino-config",
        type=Path,
        default=_REPO_ROOT / "configs/dino_dbt.yaml",
        help="Must match SSL run (image_size, num_prototypes, TinyViT dims).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = full train split")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 = full val split")
    parser.add_argument("--shuffle-labels", action="store_true", help="Permutation test: shuffle train labels.")
    parser.add_argument(
        "--data-repo-root",
        type=Path,
        default=Path("../dbt_simclr_project"),
        help="Repo containing artifacts/manifests and data config",
    )
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (_REPO_ROOT / args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    amp = (not args.no_amp) and device.type == "cuda"

    dino_cfg_path = args.dino_config if args.dino_config.is_absolute() else (_REPO_ROOT / args.dino_config).resolve()
    model_yaml = _load_dino_model_section(dino_cfg_path)
    image_size = int(model_yaml.get("image_size", 224))
    num_prototypes = int(model_yaml.get("num_prototypes", 512))

    data_root = args.data_repo_root if args.data_repo_root.is_absolute() else (_REPO_ROOT / args.data_repo_root).resolve()
    data_cfg = yaml.safe_load((data_root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    artifacts = data_root / data_cfg.get("artifacts_dir", "artifacts")
    manifest_path = artifacts / data_cfg.get("manifest_rel_path", "manifests/master_manifest.parquet")
    split_path = artifacts / data_cfg.get("split_rel_path", "splits/patient_split_v1.json")
    use_cache = bool(data_cfg.get("use_processed_cache", False))
    cache_dir = Path(data_cfg.get("processed_cache_rel_path", "")) if use_cache else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = data_root / cache_dir

    ds_cfg = SupervisedSliceConfig(
        resize_height=image_size,
        resize_width=image_size,
        normalize=bool(data_cfg.get("normalize", True)),
        split_seed=int(args.seed),
        use_processed_cache=use_cache,
        processed_cache_dir=cache_dir,
        processed_cache_token=str(data_cfg.get("processed_cache_token", "")),
    )
    train_ds = DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path, split="train", config=ds_cfg)
    val_ds = DBTSupervisedSliceDataset(manifest_path=manifest_path, split_path=split_path, split="val", config=ds_cfg)
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_supervised,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=(int(args.max_val_batches) > 0),
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_supervised,
        generator=g,
    )

    ssl = create_dino_ssl(
        image_size=image_size,
        num_prototypes=num_prototypes,
        embed_dim=int(model_yaml.get("embed_dim", 192)),
        depth=int(model_yaml.get("depth", 4)),
        num_heads=int(model_yaml.get("num_heads", 3)),
        head_hidden_dim=int(model_yaml.get("head_hidden_dim", 512)),
        head_bottleneck_dim=int(model_yaml.get("head_bottleneck_dim", 256)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        ssl.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError:
        ssl.load_state_dict(_normalize_ddp_state_dict_keys(ckpt["model_state"]), strict=True)

    max_train_batches = int(args.max_train_batches) if int(args.max_train_batches) > 0 else None
    max_val_batches = int(args.max_val_batches) if int(args.max_val_batches) > 0 else None
    Xtr, ytr = _extract_cls_embeddings(ssl.student_backbone, train_loader, device, amp=amp, max_batches=max_train_batches)
    Xva, yva = _extract_cls_embeddings(ssl.student_backbone, val_loader, device, amp=amp, max_batches=max_val_batches)
    if args.shuffle_labels and len(ytr):
        rng = np.random.default_rng(int(args.seed))
        ytr = rng.permutation(ytr)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        solver="lbfgs",
        class_weight=None,
    )
    clf.fit(Xtr, ytr)
    p_va = clf.predict_proba(Xva)[:, 1] if len(Xva) else np.zeros((0,), dtype=np.float64)
    y_hat = (p_va >= 0.5).astype(np.int64) if len(p_va) else np.zeros((0,), dtype=np.int64)
    auroc = float(roc_auc_score(yva, p_va)) if len(np.unique(yva)) >= 2 else float("nan")
    bal_acc = float(balanced_accuracy_score(yva, y_hat)) if len(yva) else float("nan")

    print(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "n_train": int(len(ytr)),
                "n_val": int(len(yva)),
                "train_pos_frac": float((ytr == 1).mean()) if len(ytr) else float("nan"),
                "val_pos_frac": float((yva == 1).mean()) if len(yva) else float("nan"),
                "val_auroc": auroc,
                "val_balanced_accuracy@0.5": bal_acc,
                "shuffle_labels": bool(args.shuffle_labels),
                "embed_dim": int(Xtr.shape[1]) if Xtr.ndim == 2 else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

