"""Student–teacher ViT + DINO head (phase 1: image-level loss only)."""

from __future__ import annotations

import copy
import torch
from torch import nn

from mammodino_ssl.models.toy_vit import TinyViT


class DINOHead(nn.Module):
    """Bottleneck MLP then prototype logits (K-way)."""

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DINOSSLModule(nn.Module):
    """Student and teacher (EMA) share architecture; teacher does not receive gradients."""

    def __init__(
        self,
        *,
        backbone: TinyViT,
        num_prototypes: int,
        num_patch_prototypes: int | None = None,
        head_hidden_dim: int = 512,
        head_bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        d = backbone.embed_dim
        p_out = int(num_patch_prototypes or num_prototypes)
        self.student_backbone = backbone
        self.student_head = DINOHead(d, head_hidden_dim, head_bottleneck_dim, num_prototypes)
        self.student_patch_head = DINOHead(d, head_hidden_dim, head_bottleneck_dim, p_out)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = copy.deepcopy(self.student_head)
        self.teacher_patch_head = copy.deepcopy(self.student_patch_head)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_head.parameters():
            p.requires_grad = False
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        self.teacher_patch_head.eval()

    @staticmethod
    def _unwrap_module(m: nn.Module) -> nn.Module:
        """Return inner module when wrapped by DDP/DataParallel."""
        return m.module if hasattr(m, "module") else m

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        cls, _ = self.student_backbone(x)
        return self.student_head(cls)

    def forward_student_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-2 hook for iBOT: expose student cls + patch tokens."""
        return self.student_backbone(x)

    def forward_student_masked_tokens(self, x: torch.Tensor, patch_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Student forward with patch masking for iBOT-style training."""
        bb = self._unwrap_module(self.student_backbone)
        return bb.forward_masked(x, patch_mask)

    def forward_student_patch_logits(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Patch prototypes for student branch; patch_tokens shape (B,N,D)."""
        return self.student_patch_head(patch_tokens)

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        cls, _ = self.teacher_backbone(x)
        return self.teacher_head(cls)

    @torch.no_grad()
    def forward_teacher_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-2/3 hook: expose teacher cls + patch tokens."""
        return self.teacher_backbone(x)

    @torch.no_grad()
    def forward_teacher_patch_logits(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Patch prototypes for teacher branch; patch_tokens shape (B,N,D)."""
        return self.teacher_patch_head(patch_tokens)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        m = float(momentum)
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data = pt.data * m + ps.data * (1.0 - m)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data = pt.data * m + ps.data * (1.0 - m)
        for ps, pt in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
            pt.data = pt.data * m + ps.data * (1.0 - m)
        for bs, bt in zip(self.student_backbone.buffers(), self.teacher_backbone.buffers()):
            bt.data.copy_(bs.data)
        for bs, bt in zip(self.student_head.buffers(), self.teacher_head.buffers()):
            bt.data.copy_(bs.data)
        for bs, bt in zip(self.student_patch_head.buffers(), self.teacher_patch_head.buffers()):
            bt.data.copy_(bs.data)


def create_dino_ssl(
    *,
    image_size: int = 224,
    num_prototypes: int = 512,
    num_patch_prototypes: int | None = None,
    embed_dim: int = 192,
    depth: int = 4,
    num_heads: int = 3,
    head_hidden_dim: int = 512,
    head_bottleneck_dim: int = 256,
) -> DINOSSLModule:
    bb = TinyViT(
        image_size=image_size,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )
    return DINOSSLModule(
        backbone=bb,
        num_prototypes=num_prototypes,
        num_patch_prototypes=num_patch_prototypes,
        head_hidden_dim=head_hidden_dim,
        head_bottleneck_dim=head_bottleneck_dim,
    )
