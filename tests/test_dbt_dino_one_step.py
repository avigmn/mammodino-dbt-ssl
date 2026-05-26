"""Finite one-step DINO pass on a real DBT batch (if artifacts exist)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from mammodino_ssl.data import DBTDINODataset, DINODataConfig
from mammodino_ssl.models.dino_ssl import create_dino_ssl
from mammodino_ssl.train import DINOLogitCenter, collate_dino_views, dino_cross_entropy


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2] / "dbt_simclr_project"
    manifest = root / "artifacts" / "manifests" / "master_manifest.parquet"
    split = root / "artifacts" / "splits" / "patient_split_v1.json"
    return manifest, split


def test_one_step_finite_loss() -> None:
    manifest, split = _paths()
    if not manifest.exists() or not split.exists():
        pytest.skip("DBT manifest/split not available for one-step smoke test")
    try:
        import pandas  # noqa: F401
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"pandas unavailable in current env: {type(e).__name__}")

    cfg = DINODataConfig(
        resize_height=384,
        resize_width=384,
        normalize=True,
        split_seed=42,
        use_processed_cache=False,
        stochastic_val_views=False,
    )
    ds = DBTDINODataset(manifest_path=manifest, split_path=split, split="train", config=cfg)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_dino_views, drop_last=True)
    batch = next(iter(loader))
    vt = batch["view_teacher"]
    vs = batch["view_student"]

    model = create_dino_ssl(image_size=384, num_prototypes=512, embed_dim=96, depth=2, num_heads=3)
    center = DINOLogitCenter(512, center_momentum=0.9)
    with torch.no_grad():
        t_logits = model.forward_teacher(vt)
    s_logits = model.forward_student(vs)
    loss = dino_cross_entropy(s_logits, t_logits, center.center, student_temp=0.1, teacher_temp=0.04)
    assert torch.isfinite(loss).all()
