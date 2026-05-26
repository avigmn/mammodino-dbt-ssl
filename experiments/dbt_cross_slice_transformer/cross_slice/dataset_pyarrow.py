"""Supervised slice dataset using pyarrow manifest reads (no pandas).

Mirrors ``DBTSupervisedSliceDataset`` IO behavior without importing ``dbt_ssl.data.dbt_dataset``
(which pulls pandas). Uses existing numpy/PIL/cache helpers from ``dbt_ssl``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq

from dbt_ssl.data.artifact_stress import apply_artifact_stress
from dbt_ssl.data.io_png import load_png_grayscale
from dbt_ssl.data.preprocess import normalize_to_unit_range, resize_array
from dbt_ssl.data.processed_cache import cache_file_path, default_cache_token, load_processed_array

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SandboxSliceConfig:
    """Subset of fields needed for embedding extraction (matches supervised probe settings)."""

    resize_height: int
    resize_width: int
    normalize: bool = True
    split_seed: int = 42
    use_processed_cache: bool = False
    processed_cache_dir: Path | None = None
    processed_cache_token: str = ""
    max_bad_sample_retries: int = 32
    stress_mode: str | None = None
    stress_blur_radius: float = 18.0
    stress_border_px: int = 18


class DBTSupervisedSliceDatasetPyArrow:
    """One row per slice; ``label_clinical`` binary; same splits as SSL manifest."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        split_path: str | Path,
        split: Split,
        config: SandboxSliceConfig,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split_path = Path(split_path)
        self.split = split
        self.config = config
        self._rng = random.Random(config.split_seed)
        if config.use_processed_cache and config.processed_cache_dir is None:
            raise ValueError("use_processed_cache=True requires processed_cache_dir")
        self._warned_bad_paths: set[str] = set()
        self._rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, object]]:
        table = pq.read_table(self.manifest_path)
        cols = table.column_names
        needed = {"canonical_patient_id", "slice_abspath", "label_clinical"}
        missing = needed - set(cols)
        if missing:
            raise ValueError(f"manifest missing columns {sorted(missing)}; have {cols}")

        split_map_raw = json.loads(self.split_path.read_text(encoding="utf-8"))
        split_map = {str(k): str(v) for k, v in split_map_raw.items()}

        pids = [str(x) for x in table.column("canonical_patient_id").to_pylist()]
        uniq_pids = sorted(set(pids))
        missing_assign = [p for p in uniq_pids if p not in split_map]
        if missing_assign:
            raise ValueError(f"manifest rows missing split assignment: {missing_assign[:10]}")

        paths = [str(x) for x in table.column("slice_abspath").to_pylist()]
        labels = table.column("label_clinical").to_pylist()

        rows: list[dict[str, object]] = []
        for i in range(table.num_rows):
            pid = pids[i]
            sp = split_map[pid]
            if sp != self.split:
                continue
            lab = int(labels[i])
            rows.append(
                {
                    "canonical_patient_id": pid,
                    "slice_abspath": paths[i],
                    "label_clinical": lab,
                }
            )

        return rows

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        self._rng = random.Random(self.config.split_seed + epoch)

    def _load_base_hw(self, ap: str, token: str) -> np.ndarray:
        image = None
        if self.config.use_processed_cache and self.config.processed_cache_dir is not None:
            cache_path = cache_file_path(self.config.processed_cache_dir, ap, token)
            if cache_path.exists():
                image = load_processed_array(cache_path)
                if image.shape != (self.config.resize_height, self.config.resize_width):
                    raise ValueError(
                        f"cached shape {image.shape} != expected "
                        f"({self.config.resize_height}, {self.config.resize_width}) for {cache_path}"
                    )
        if image is None:
            image = load_png_grayscale(ap)
            image = resize_array(
                image,
                height=self.config.resize_height,
                width=self.config.resize_width,
                interpolation="bilinear",
            )
            if self.config.normalize:
                image = normalize_to_unit_range(image)
        if self.config.stress_mode:
            image = apply_artifact_stress(
                image,
                self.config.stress_mode,
                blur_radius=self.config.stress_blur_radius,
                border_px=self.config.stress_border_px,
            )
        return image

    def __getitem__(self, index: int) -> dict:
        token = self.config.processed_cache_token or default_cache_token(
            self.config.resize_height,
            self.config.resize_width,
            self.config.normalize,
        )
        n = len(self._rows)
        if n == 0:
            raise IndexError("Dataset is empty")

        retries = max(1, self.config.max_bad_sample_retries)
        for attempt in range(retries):
            row_idx = (index + attempt) % n
            row = self._rows[row_idx]
            ap = str(row["slice_abspath"])
            try:
                base = self._load_base_hw(ap, token)
                image = np.expand_dims(base.astype(np.float32), axis=0)
                lab = int(row["label_clinical"])
                if lab not in (0, 1):
                    raise ValueError(f"expected binary label_clinical, got {lab}")
                return {
                    "image": image,
                    "label": lab,
                    "canonical_patient_id": row["canonical_patient_id"],
                    "slice_abspath": row["slice_abspath"],
                }
            except Exception as e:
                if ap not in self._warned_bad_paths:
                    self._warned_bad_paths.add(ap)
                    print(f"[WARN] Skipping unreadable sample: {ap} ({type(e).__name__})")
                continue

        raise RuntimeError(
            f"Failed to fetch a valid sample after {retries} retries from index {index}. "
            "Too many unreadable samples."
        )
