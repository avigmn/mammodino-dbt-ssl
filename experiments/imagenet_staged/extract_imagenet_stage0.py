#!/usr/bin/env python3
"""Produce Stage-0-format embeddings from a frozen ImageNet ViT-tiny.

Reuses Liron's stage0_extract_embeddings helpers (metadata build + assembly) so
the output parquet schema is byte-identical to her DINO-full Stage 0 — only the
backbone is swapped to a frozen timm `vit_tiny_patch16_224` (ImageNet weights),
with ImageNet mean/std normalization applied inside forward.

Output (writable area): /mnt/data/avi/imagenet_staged/stage_0_embeddings/
  frozen_dino_embeddings_{train,val,test,all}.parquet  (12 meta cols + emb_cls_final, 192-d)

Then Liron's heads can be run on it via --stage0-dir / --embeddings-dir.

Run:
  PY=/mnt/md0/Liron/mammodino_ssl_project/.venv/bin/python
  PYTHONPATH=/mnt/data/avi/py_packages $PY extract_imagenet_stage0.py
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

MAMMO = Path("/mnt/md0/Liron/mammodino_ssl_project_dino_full")
DBT = Path("/mnt/md0/Liron/dbt_simclr_project")
for p in (str(MAMMO / "src"), str(DBT / "src"), "/mnt/data/avi/py_packages"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402

# import Liron's stage0 module by path (reuse her exact metadata + assembly)
_spec = importlib.util.spec_from_file_location(
    "stage0", MAMMO / "scripts/head_evaluation/stage0_extract_embeddings.py"
)
stage0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage0)

OUT_DIR = Path("/mnt/data/avi/imagenet_staged/stage_0_embeddings")
MODEL_NAME = "vit_tiny_patch16_224"


class ImageNetBackbone(torch.nn.Module):
    """Frozen ImageNet ViT-tiny; input [B,3,H,W] in [0,1] -> 192-d CLS."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.embed_dim = int(self.model.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)  # [B, 192]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("imagenet_stage0")

    paths = stage0._load_data_paths(DBT)
    manifest_path = paths["manifest_path"]
    split_path = paths["split_path"]
    meta = stage0._meta_columns(stage0._build_metadata_table(manifest_path, split_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = ImageNetBackbone().to(device).eval()
    logger.info("Loaded %s (embed_dim=%d) on %s", MODEL_NAME, backbone.embed_dim, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("train", "val", "test"):
        logger.info("=== extracting %s ===", split)
        paths_o, pids_o, labels, variant_arrays = stage0._extract_split_embeddings(
            backbone=backbone,
            manifest_path=manifest_path,
            split_path=split_path,
            split=split,
            image_size=224,
            normalize=True,           # dataset min-max [0,1]; ImageNet norm applied in backbone
            seed=42,
            batch_size=256,
            num_workers=8,
            device=device,
            amp=True,
            variants=["cls_final"],
            max_batches=None,
            logger=logger,
        )
        df = stage0._assemble_split_frame(
            split=split,
            paths=paths_o,
            pids=pids_o,
            labels=labels,
            variant_arrays=variant_arrays,
            meta=meta,
            logger=logger,
        )
        df.to_parquet(OUT_DIR / f"frozen_dino_embeddings_{split}.parquet")
        frames.append(df)
        logger.info("  wrote %s rows=%d", split, len(df))

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(OUT_DIR / "frozen_dino_embeddings_all.parquet")
    logger.info("DONE. total rows=%d -> %s", len(all_df), OUT_DIR)


if __name__ == "__main__":
    main()
