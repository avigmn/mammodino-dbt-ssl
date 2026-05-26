"""DBT DINO augmentations: weak teacher view and strong student view."""

from __future__ import annotations

import torchvision.transforms as T


def make_teacher_dino_transform(
    *,
    output_hw: tuple[int, int],
    rrc_scale: tuple[float, float] = (0.85, 1.0),
    rrc_ratio: tuple[float, float] = (0.9, 1.1),
    horizontal_flip_p: float = 0.5,
) -> T.Compose:
    """Weak/stable teacher transform for 1-channel [0,1] tensors."""
    h, w = output_hw
    return T.Compose(
        [
            T.RandomResizedCrop((h, w), scale=rrc_scale, ratio=rrc_ratio, antialias=True),
            T.RandomHorizontalFlip(p=horizontal_flip_p),
        ]
    )


def make_student_dino_transform(
    *,
    output_hw: tuple[int, int],
    rrc_scale: tuple[float, float] = (0.2, 1.0),
    rrc_ratio: tuple[float, float] = (0.75, 1.3333333333),
    horizontal_flip_p: float = 0.5,
    color_jitter_p: float = 0.8,
    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    gaussian_blur_p: float = 0.5,
    gaussian_blur_kernel: int = 23,
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0),
) -> T.Compose:
    """Strong student transform for 1-channel [0,1] tensors."""
    h, w = output_hw
    k = gaussian_blur_kernel if gaussian_blur_kernel % 2 == 1 else gaussian_blur_kernel + 1
    jitter = T.ColorJitter(
        brightness=color_jitter_brightness,
        contrast=color_jitter_contrast,
        saturation=0.0,
        hue=0.0,
    )
    return T.Compose(
        [
            T.RandomResizedCrop((h, w), scale=rrc_scale, ratio=rrc_ratio, antialias=True),
            T.RandomHorizontalFlip(p=horizontal_flip_p),
            T.RandomApply([jitter], p=color_jitter_p),
            T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=gaussian_blur_sigma)], p=gaussian_blur_p),
        ]
    )


def make_eval_dino_transform(output_hw: tuple[int, int]) -> T.Compose:
    """Deterministic val/test transform for already-resized tensors."""
    h, w = output_hw
    return T.Compose([T.CenterCrop((h, w))]) if h > 0 and w > 0 else T.Compose([T.Lambda(lambda x: x)])
