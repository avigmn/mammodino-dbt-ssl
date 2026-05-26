"""DINO image-level loss + EMA centering on teacher logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DINOLogitCenter(nn.Module):
    """EMA of mean teacher logits per batch (shape (1, K))."""

    def __init__(self, num_prototypes: int, center_momentum: float = 0.9) -> None:
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, num_prototypes))

    @torch.no_grad()
    def update_from_teacher_logits(self, teacher_logits: torch.Tensor) -> None:
        """teacher_logits: (B, K), detached."""
        batch_mean = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_mean, alpha=1.0 - self.center_momentum)


def dino_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    center: torch.Tensor,
    *,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    """Negative cross-entropy between softmax(teacher) and log_softmax(student), with center on teacher."""
    t = (teacher_logits - center) / float(teacher_temp)
    p = F.softmax(t, dim=-1)
    s = F.log_softmax(student_logits / float(student_temp), dim=-1)
    return -(p * s).sum(dim=-1).mean()


def masked_patch_cross_entropy(
    student_patch_logits: torch.Tensor,
    teacher_patch_logits: torch.Tensor,
    patch_mask: torch.Tensor,
    center: torch.Tensor,
    *,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    """
    iBOT-style CE over masked patch tokens only.

    Args:
        student_patch_logits: (B, N, K)
        teacher_patch_logits: (B, N, K)
        patch_mask: (B, N) bool, True for masked tokens
        center: (1, K) teacher-logit center
    """
    if student_patch_logits.shape != teacher_patch_logits.shape:
        raise ValueError(
            f"student/teacher patch logits shape mismatch: "
            f"{tuple(student_patch_logits.shape)} vs {tuple(teacher_patch_logits.shape)}"
        )
    if patch_mask.ndim != 2 or patch_mask.shape[:2] != student_patch_logits.shape[:2]:
        raise ValueError(
            f"patch_mask must be (B,N) matching patch logits; got {tuple(patch_mask.shape)} "
            f"for logits {tuple(student_patch_logits.shape)}"
        )
    mask = patch_mask.bool()
    if not bool(mask.any()):
        # Return zero-like scalar to keep graph valid and numerically stable.
        return student_patch_logits.sum() * 0.0
    t = (teacher_patch_logits - center.unsqueeze(1)) / float(teacher_temp)
    p = F.softmax(t, dim=-1)
    s = F.log_softmax(student_patch_logits / float(student_temp), dim=-1)
    per_patch = -(p * s).sum(dim=-1)  # (B, N)
    return per_patch[mask].mean()
