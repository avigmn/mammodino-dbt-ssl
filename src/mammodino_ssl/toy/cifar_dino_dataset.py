"""CIFAR-10: weak global view for teacher, strong global view for student (two crops, no small crops)."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset
from torchvision import transforms

from dbt_ssl.toy.cifar_simclr_dataset import (
    CIFAR_MEAN,
    CIFAR_STD,
    RobustCIFAR10,
    build_simclr_train_transform,
)


def build_teacher_global_transform(*, image_size: int = 224) -> transforms.Compose:
    """Milder augmentations for the teacher branch."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


class CIFAR10DINOPairDataset(Dataset):
    """One weak and one strong view per image (labels unused during SSL)."""

    def __init__(self, root: str, *, train: bool, image_size: int = 224) -> None:
        self._base = RobustCIFAR10(root=root, train=train, download=train, transform=None)
        if train:
            self._teacher_tfm = build_teacher_global_transform(image_size=image_size)
            self._student_tfm = build_simclr_train_transform(image_size=image_size)
        else:
            eval_tfm = transforms.Compose(
                [
                    transforms.Resize(image_size, antialias=True),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )
            self._teacher_tfm = eval_tfm
            self._student_tfm = eval_tfm

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pil, y = self._base[index]
        return {
            "view_teacher": self._teacher_tfm(pil),
            "view_student": self._student_tfm(pil),
            "label": int(y),
        }


class CIFAR10SupervisedEvalDataset(Dataset):
    """Deterministic single view for linear probe (same as dbt toy eval)."""

    def __init__(self, root: str, *, train: bool, image_size: int = 224) -> None:
        self._base = RobustCIFAR10(root=root, train=train, download=False, transform=None)
        self._tfm = transforms.Compose(
            [
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pil, y = self._base[index]
        return {"image": self._tfm(pil), "label": int(y)}
