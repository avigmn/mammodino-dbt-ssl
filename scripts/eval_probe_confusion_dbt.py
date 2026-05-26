#!/usr/bin/env python3
"""DBT linear probe evaluation: confusion matrix + ROC (CLI; logic in mammodino_ssl.eval)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from mammodino_ssl.eval import save_confusion_roc_metrics
from mammodino_ssl.models.dino_ssl import create_dino_ssl
from mammodino_ssl.models.linear_probe import FrozenTinyViTLinearProbe


def _load_dino_model_section(config_path: Path) -> dict:
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return dict(raw.get("model") or {})


def _to_tensor3(image_nchw: object) -> torch.Tensor:
    import numpy as np

    t = torch.from_numpy(image_nchw).float()  # type: ignore[arg-type]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion matrix + ROC from DINO + trained linear probe.")
    parser.add_argument("--dino-checkpoint", required=True, type=Path, help="DINO best.pt (model_state)")
    parser.add_argument("--probe-checkpoint", required=True, type=Path, help="probe_best.pt (head_state)")
    parser.add_argument(
        "--probe-head",
        choices=["linear", "mlp"],
        default="linear",
        help="Probe head type used during training (must match checkpoint).",
    )
    parser.add_argument(
        "--probe-mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden dim for MLP head (when --probe-head mlp).",
    )
    parser.add_argument(
        "--probe-dropout",
        type=float,
        default=0.0,
        help="Dropout for MLP head (when --probe-head mlp).",
    )
    parser.add_argument(
        "--dino-config",
        type=Path,
        default=_REPO_ROOT / "configs/dino_dbt.yaml",
        help="Must match SSL run (image_size, num_prototypes, TinyViT dims).",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--positive-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full split")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: probe_checkpoint_dir/confusion_eval)",
    )
    parser.add_argument(
        "--data-repo-root",
        type=Path,
        default=Path("../dbt_simclr_project"),
        help="Repo containing artifacts/manifests and data config",
    )
    args = parser.parse_args()

    dino_ckpt_path = args.dino_checkpoint if args.dino_checkpoint.is_absolute() else (_REPO_ROOT / args.dino_checkpoint).resolve()
    probe_ckpt_path = args.probe_checkpoint if args.probe_checkpoint.is_absolute() else (_REPO_ROOT / args.probe_checkpoint).resolve()
    if not dino_ckpt_path.is_file():
        raise FileNotFoundError(f"DINO checkpoint not found: {dino_ckpt_path}")
    if not probe_ckpt_path.is_file():
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_ckpt_path}")

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
        split_seed=42,
        use_processed_cache=use_cache,
        processed_cache_dir=cache_dir,
        processed_cache_token=str(data_cfg.get("processed_cache_token", "")),
    )
    ds = DBTSupervisedSliceDataset(
        manifest_path=manifest_path,
        split_path=split_path,
        split=args.split,
        config=ds_cfg,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_supervised,
    )

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    amp = (not args.no_amp) and device.type == "cuda"

    ssl = create_dino_ssl(
        image_size=image_size,
        num_prototypes=num_prototypes,
        embed_dim=int(model_yaml.get("embed_dim", 192)),
        depth=int(model_yaml.get("depth", 4)),
        num_heads=int(model_yaml.get("num_heads", 3)),
        head_hidden_dim=int(model_yaml.get("head_hidden_dim", 512)),
        head_bottleneck_dim=int(model_yaml.get("head_bottleneck_dim", 256)),
    )
    dino_ckpt = torch.load(dino_ckpt_path, map_location=device, weights_only=False)
    ssl.load_state_dict(dino_ckpt["model_state"], strict=True)
    probe = FrozenTinyViTLinearProbe(
        ssl.student_backbone,
        num_classes=2,
        head_type=args.probe_head,
        mlp_hidden_dim=args.probe_mlp_hidden_dim,
        dropout=args.probe_dropout,
    ).to(device)
    probe_ckpt = torch.load(probe_ckpt_path, map_location=device, weights_only=False)
    probe.head.load_state_dict(probe_ckpt["head_state"], strict=True)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = probe_ckpt_path.parent / "confusion_eval"
    elif not out_dir.is_absolute():
        out_dir = (_REPO_ROOT / out_dir).resolve()

    max_steps = args.max_steps if args.max_steps > 0 else None
    summary = save_confusion_roc_metrics(
        probe,
        loader,
        device=device,
        amp=amp,
        out_dir=out_dir,
        split=args.split,
        positive_class=args.positive_class,
        max_steps=max_steps,
        dino_checkpoint=str(dino_ckpt_path),
        probe_checkpoint=str(probe_ckpt_path),
    )
    paths = summary.pop("_paths", {})
    print(json.dumps({"out_dir": str(out_dir.resolve()), **paths, **{k: v for k, v in summary.items() if k not in ("fpr", "tpr")}}, indent=2))


if __name__ == "__main__":
    main()
