"""Smoke checks for DBT DINO dataset wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from mammodino_ssl.data import DBTDINODataset, DINODataConfig


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2] / "dbt_simclr_project"
    manifest = root / "artifacts" / "manifests" / "master_manifest.parquet"
    split = root / "artifacts" / "splits" / "patient_split_v1.json"
    return manifest, split


def test_dataset_returns_expected_keys_and_shapes() -> None:
    manifest, split = _paths()
    if not manifest.exists() or not split.exists():
        pytest.skip("DBT manifest/split not available for smoke dataset test")
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
    sample = ds[0]
    assert "view_teacher" in sample
    assert "view_student" in sample
    assert tuple(sample["view_teacher"].shape) == (3, 384, 384)
    assert tuple(sample["view_student"].shape) == (3, 384, 384)
    assert sample["future_patch_mask"] is None
    assert sample["future_neighbor_view"] is None
