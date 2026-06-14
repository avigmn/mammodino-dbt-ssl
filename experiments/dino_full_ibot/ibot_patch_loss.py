"""iBOT masked-patch loss + tissue-aware mask sampling (additive fork).

Cross-entropy between the student's predictions at MASKED patch positions and
the teacher's centered+sharpened patch distribution at the same positions, on
the global crops. Mirrors the DINO centering/temperature mechanics at the patch
level (own center buffer + teacher-temp warmup).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mammodino_ssl.train.schedules import teacher_temp_schedule


class iBOTPatchLoss(nn.Module):
    def __init__(
        self,
        patch_out_dim: int,
        *,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        nepochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.teacher_temp_schedule = teacher_temp_schedule(
            warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs
        )

    def forward(
        self,
        student_patch: torch.Tensor,   # (gB, N, Kp) at all positions
        teacher_patch: torch.Tensor,   # (gB, N, Kp) at all positions (no grad)
        mask: torch.Tensor,            # (gB, N) bool: True = masked (predicted)
        epoch: int,
    ) -> torch.Tensor:
        epoch = int(min(max(epoch, 0), len(self.teacher_temp_schedule) - 1))
        temp = float(self.teacher_temp_schedule[epoch])
        t = F.softmax((teacher_patch - self.center) / temp, dim=-1).detach()
        s = F.log_softmax(student_patch / self.student_temp, dim=-1)
        loss = torch.sum(-t * s, dim=-1)  # (gB, N)
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        out = (loss * m).sum() / denom
        self.update_center(teacher_patch, mask)
        return out

    @torch.no_grad()
    def update_center(self, teacher_patch: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.unsqueeze(-1).float()
        summed = (teacher_patch * m).sum(dim=(0, 1), keepdim=True)  # (1,1,Kp)
        count = m.sum().clamp_min(1.0)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(summed)
            cnt = torch.tensor([count], device=summed.device)
            dist.all_reduce(cnt)
            count = cnt.item()
        batch_center = summed / count
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)


def sample_patch_masks(
    tissue_patch_weights: torch.Tensor | None,
    *,
    n_global: int,
    batch: int,
    n_patches: int,
    mask_ratio: float,
    tissue_bias: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a (n_global*batch, n_patches) bool mask.

    Each global crop of each sample masks ~mask_ratio of its patches. Selection
    is biased toward tissue patches: score = (1-bias)*uniform + bias*tissue_w.
    Falls back to uniform if no tissue weights are provided.
    """
    device = tissue_patch_weights.device if tissue_patch_weights is not None else torch.device("cpu")
    k = max(1, int(round(mask_ratio * n_patches)))
    gB = n_global * batch
    if tissue_patch_weights is not None and tissue_patch_weights.shape[-1] == n_patches:
        # (B, N) -> tile across global crops -> (gB, N)
        tw = tissue_patch_weights.to(device).float()
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        tw = tw.repeat(n_global, 1)
    else:
        tw = torch.full((gB, n_patches), 1.0 / n_patches, device=device)
    unif = torch.rand(gB, n_patches, device=device, generator=generator)
    score = (1.0 - tissue_bias) * unif + tissue_bias * tw
    topk = score.topk(k, dim=-1).indices
    mask = torch.zeros(gB, n_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, topk, True)
    return mask
