"""DBT DINO dataset over manifest + patient split with weak/strong views."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from dbt_ssl.data.artifact_stress import apply_artifact_stress
from dbt_ssl.data.io_png import load_png_grayscale
from dbt_ssl.data.preprocess import normalize_to_unit_range, resize_array
from dbt_ssl.data.processed_cache import cache_file_path, default_cache_token, load_processed_array
from mammodino_ssl.data.augmentations_dbt_dino import (
    make_eval_dino_transform,
    make_student_dino_transform,
    make_teacher_dino_transform,
)

Split = Literal["train", "val", "test"]
VolumePairMode = Literal["off", "adjacent"]


@dataclass(frozen=True)
class DINODataConfig:
    """DBT DINO config: same IO semantics as SimCLR with teacher/student transforms."""

    resize_height: int = 384
    resize_width: int = 384
    normalize: bool = True
    split_seed: int = 42
    use_processed_cache: bool = False
    processed_cache_dir: Path | None = None
    processed_cache_token: str = ""
    max_bad_sample_retries: int = 32
    stress_mode: str | None = None
    stress_blur_radius: float = 18.0
    stress_border_px: int = 18
    volume_pair_mode: VolumePairMode = "off"
    # Teacher (weak)
    teacher_rrc_scale_min: float = 0.85
    teacher_rrc_scale_max: float = 1.0
    teacher_horizontal_flip_p: float = 0.5
    # Student (strong)
    student_rrc_scale_min: float = 0.2
    student_rrc_scale_max: float = 1.0
    student_horizontal_flip_p: float = 0.5
    student_color_jitter_p: float = 0.8
    student_color_jitter_brightness: float = 0.2
    student_color_jitter_contrast: float = 0.2
    student_gaussian_blur_p: float = 0.5
    student_gaussian_blur_kernel: int = 23
    # If True, val/test are stochastic; else deterministic center crop on both branches.
    stochastic_val_views: bool = False
    # Tissue-aware foreground mask settings.
    tissue_percentile: float = 80.0
    tissue_mask_cleanup: bool = True
    tissue_crop_bias_prob: float = 0.6
    tissue_crop_pad_frac: float = 0.08
    patch_size: int = 16


class DBTDINODataset:
    """Two views per sample: teacher weak and student strong."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        split_path: str | Path,
        split: Split,
        config: DINODataConfig,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split_path = Path(split_path)
        self.split = split
        self.config = config
        self._rng = random.Random(config.split_seed)
        self._df = self._load_index()
        if config.use_processed_cache and config.processed_cache_dir is None:
            raise ValueError("use_processed_cache=True requires processed_cache_dir")
        self._warned_bad_paths: set[str] = set()

        out_hw = (config.resize_height, config.resize_width)
        use_stochastic = split == "train" or config.stochastic_val_views
        if use_stochastic:
            self._teacher_aug = make_teacher_dino_transform(
                output_hw=out_hw,
                rrc_scale=(config.teacher_rrc_scale_min, config.teacher_rrc_scale_max),
                horizontal_flip_p=config.teacher_horizontal_flip_p,
            )
            self._student_aug = make_student_dino_transform(
                output_hw=out_hw,
                rrc_scale=(config.student_rrc_scale_min, config.student_rrc_scale_max),
                horizontal_flip_p=config.student_horizontal_flip_p,
                color_jitter_p=config.student_color_jitter_p,
                color_jitter_brightness=config.student_color_jitter_brightness,
                color_jitter_contrast=config.student_color_jitter_contrast,
                gaussian_blur_p=config.student_gaussian_blur_p,
                gaussian_blur_kernel=config.student_gaussian_blur_kernel,
            )
        else:
            eval_aug = make_eval_dino_transform(out_hw)
            self._teacher_aug = eval_aug
            self._student_aug = eval_aug

        self._adjacent_path: dict[str, str | None] = {}
        if config.volume_pair_mode == "adjacent":
            self._adjacent_path = self._build_adjacent_path_map(self._df)

    @staticmethod
    def _build_adjacent_path_map(df: object) -> dict[str, str | None]:
        if "volume_key" not in df.columns or "z_rank" not in df.columns:
            return {}
        out: dict[str, str | None] = {}
        for _, g in df.groupby("volume_key", sort=False):
            g2 = g.sort_values("z_rank")
            paths = g2["slice_abspath"].astype(str).tolist()
            for i, p in enumerate(paths):
                out[p] = paths[i + 1] if i + 1 < len(paths) else None
        return out

    def _load_index(self):
        import pandas as pd

        df = pd.read_parquet(self.manifest_path)
        split_map = json.loads(self.split_path.read_text(encoding="utf-8"))
        df = df.copy()
        df["split"] = df["canonical_patient_id"].map(split_map)
        if df["split"].isna().any():
            missing = sorted(df.loc[df["split"].isna(), "canonical_patient_id"].unique().tolist())
            raise ValueError(f"manifest rows missing split assignment: {missing[:10]}")
        return df[df["split"] == self.split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

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

    @staticmethod
    def _to_tensor_3chw(hw: np.ndarray) -> torch.Tensor:
        """Convert HxW [0,1] to 3xHxW float tensor for TinyViT input."""
        t = torch.from_numpy(hw.astype(np.float32)).unsqueeze(0).clamp(0.0, 1.0)
        return t.repeat(3, 1, 1)

    @staticmethod
    def _minmax_01(hw: np.ndarray) -> np.ndarray:
        x = hw.astype(np.float32, copy=False)
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
        return (x - lo) / (hi - lo)

    def _build_tissue_mask(self, hw: np.ndarray) -> np.ndarray:
        x = self._minmax_01(hw)
        thr = float(np.percentile(x, self.config.tissue_percentile))
        mask = (x >= thr).astype(np.uint8)
        if self.config.tissue_mask_cleanup:
            try:
                from scipy import ndimage as ndi

                mask = ndi.binary_opening(mask, structure=np.ones((3, 3))).astype(np.uint8)
                mask = ndi.binary_closing(mask, structure=np.ones((5, 5))).astype(np.uint8)
            except Exception:
                # Fallback without scipy: light cleanup via neighbor count.
                m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                k = torch.ones((1, 1, 3, 3), dtype=torch.float32)
                nb = torch.nn.functional.conv2d(m, k, padding=1)
                mask = ((nb >= 3.0).squeeze().numpy()).astype(np.uint8)
        if int(mask.sum()) == 0:
            # Never return empty mask; fallback to all-foreground.
            mask = np.ones_like(mask, dtype=np.uint8)
        return mask

    def _downsample_mask_to_patch_grid(self, mask_hw: np.ndarray) -> torch.Tensor:
        h, w = mask_hw.shape
        p = int(self.config.patch_size)
        gh = max(1, h // p)
        gw = max(1, w // p)
        hh = gh * p
        ww = gw * p
        m = mask_hw[:hh, :ww].reshape(gh, p, gw, p).mean(axis=(1, 3))  # (gh, gw) in [0,1]
        flat = m.reshape(-1).astype(np.float32)
        # Keep a positive floor so every patch remains sampleable.
        flat = flat + 1e-3
        flat = flat / float(flat.sum())
        return torch.from_numpy(flat)

    def _focus_crop_toward_tissue(self, t3: torch.Tensor, mask_hw: np.ndarray) -> torch.Tensor:
        if self.split != "train":
            return t3
        if self._rng.random() > float(self.config.tissue_crop_bias_prob):
            return t3
        ys, xs = np.where(mask_hw > 0)
        if ys.size == 0 or xs.size == 0:
            return t3
        h, w = mask_hw.shape
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad_h = int(max(1, round(h * float(self.config.tissue_crop_pad_frac))))
        pad_w = int(max(1, round(w * float(self.config.tissue_crop_pad_frac))))
        y0 = max(0, y0 - pad_h)
        y1 = min(h, y1 + pad_h)
        x0 = max(0, x0 - pad_w)
        x1 = min(w, x1 + pad_w)
        if y1 <= y0 or x1 <= x0:
            return t3
        crop = t3[:, y0:y1, x0:x1].unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            crop,
            size=(self.config.resize_height, self.config.resize_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def __getitem__(self, index: int) -> dict:
        token = self.config.processed_cache_token or default_cache_token(
            self.config.resize_height,
            self.config.resize_width,
            self.config.normalize,
        )
        n = len(self._df)
        if n == 0:
            raise IndexError("Dataset is empty")

        retries = max(1, self.config.max_bad_sample_retries)
        for attempt in range(retries):
            row_idx = (index + attempt) % n
            row = self._df.iloc[row_idx]
            ap = str(row["slice_abspath"])
            try:
                base = self._load_base_hw(ap, token)
                tissue_mask = self._build_tissue_mask(base)
                t0 = self._to_tensor_3chw(base)
                t0 = self._focus_crop_toward_tissue(t0, tissue_mask)

                second_path: str | None = None
                if self.config.volume_pair_mode == "adjacent" and self._adjacent_path:
                    second_path = self._adjacent_path.get(ap)

                if second_path is not None:
                    base_b = self._load_base_hw(second_path, token)
                    tissue_mask_b = self._build_tissue_mask(base_b)
                    t1 = self._to_tensor_3chw(base_b)
                    t1 = self._focus_crop_toward_tissue(t1, tissue_mask_b)
                    view_teacher = self._teacher_aug(t0)
                    view_student = self._student_aug(t1)
                else:
                    view_teacher = self._teacher_aug(t0)
                    view_student = self._student_aug(t0)

                out = {
                    "view_teacher": view_teacher.contiguous(),
                    "view_student": view_student.contiguous(),
                    "canonical_patient_id": row["canonical_patient_id"],
                    "source_label": row.get("source_label", ""),
                    "slice_abspath": row["slice_abspath"],
                    "volume_pair_used": bool(second_path is not None),
                    "tissue_patch_weights": self._downsample_mask_to_patch_grid(tissue_mask),
                }
                return out
            except Exception as e:
                if ap not in self._warned_bad_paths:
                    self._warned_bad_paths.add(ap)
                    print(f"[WARN] Skipping unreadable sample: {ap} ({type(e).__name__})")
                continue

        raise RuntimeError(
            f"Failed to fetch a valid sample after {retries} retries from index {index}. "
            "Too many unreadable samples."
        )

    @property
    def dataframe(self):
        return self._df.copy()
