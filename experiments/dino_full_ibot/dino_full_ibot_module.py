"""DINO-full + iBOT student/teacher module (additive fork; single-GPU).

Extends the faithful DINO-full design with an iBOT-style masked-patch branch:
  * student/teacher share the same ViT backbone (as DINO-full),
  * a CLS DINOHead (image-level DINO objective), and
  * a separate patch DINOHead (iBOT masked-patch objective).

The student forward runs BOTH paths so all student params receive gradients in a
single backward (kept single-GPU here to avoid DDP unused-param subtleties).
Teacher is EMA, no-grad, initialized from the student.

Reuses Liron's `vit.py` (forward / forward_features / forward_masked) and
`dino_head.py` (DINOHead) unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from mammodino_ssl.models.dino_head import DINOHead
from mammodino_ssl.models.vit import VisionTransformer, create_vit


class _StudentNet(nn.Module):
    """Backbone + CLS head + (optional) patch head; one forward does both paths."""

    def __init__(self, backbone: VisionTransformer, cls_head: DINOHead, patch_head: DINOHead | None) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self.patch_head = patch_head

    def forward(
        self,
        crops: list[torch.Tensor],
        n_global: int,
        patch_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (cls_out [sum_crops*B, K], patch_out [n_global*B, N, Kp] | None).

        CLS path mirrors DINO MultiCropWrapper: one backbone pass per resolution
        group, head on concatenated CLS. iBOT path: masked forward on the global
        crops only, patch head on all patch tokens (loss selects masked ones).
        """
        # --- CLS path (all crops, grouped by resolution) ---
        widths = torch.tensor([c.shape[-1] for c in crops])
        idx = torch.cumsum(torch.unique_consecutive(widths, return_counts=True)[1], dim=0)
        start = 0
        cls_cat: torch.Tensor | None = None
        for end in idx:
            chunk = torch.cat(crops[start:int(end)])
            cls = self.backbone(chunk)
            if isinstance(cls, tuple):
                cls = cls[0]
            cls_cat = cls if cls_cat is None else torch.cat((cls_cat, cls))
            start = int(end)
        assert cls_cat is not None
        cls_out = self.cls_head(cls_cat)

        # --- iBOT path (global crops only, masked) ---
        patch_out = None
        if self.patch_head is not None and patch_mask is not None:
            g = torch.cat(crops[:n_global])  # (n_global*B, C, H, W)
            _, patch_tokens = self.backbone.forward_masked(g, patch_mask)  # (gB, N, D)
            gB, N, D = patch_tokens.shape
            patch_out = self.patch_head(patch_tokens.reshape(gB * N, D)).reshape(gB, N, -1)
        return cls_out, patch_out


class _TeacherNet(nn.Module):
    def __init__(self, backbone: VisionTransformer, cls_head: DINOHead, patch_head: DINOHead | None) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self.patch_head = patch_head

    @torch.no_grad()
    def forward(
        self, global_crops: list[torch.Tensor], want_patch: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        g = torch.cat(global_crops)
        if want_patch and self.patch_head is not None:
            cls, patch_tokens = self.backbone.forward_features(g)
            gB, N, D = patch_tokens.shape
            patch_out = self.patch_head(patch_tokens.reshape(gB * N, D)).reshape(gB, N, -1)
        else:
            cls = self.backbone(g)
            if isinstance(cls, tuple):
                cls = cls[0]
            patch_out = None
        cls_out = self.cls_head(cls)
        return cls_out, patch_out


class DINOFulliBOTModule(nn.Module):
    """Student (trainable) + teacher (EMA) with CLS and patch DINO heads."""

    def __init__(
        self,
        *,
        student_backbone: VisionTransformer,
        teacher_backbone: VisionTransformer,
        out_dim: int = 4096,
        patch_out_dim: int = 4096,
        w_ibot: float = 0.1,
        head_hidden_dim: int = 2048,
        head_bottleneck_dim: int = 256,
        head_nlayers: int = 3,
        head_use_bn: bool = False,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        embed_dim = student_backbone.embed_dim
        self.w_ibot = float(w_ibot)
        use_patch = self.w_ibot > 0.0

        def _head(out: int, norm_last: bool) -> DINOHead:
            return DINOHead(
                embed_dim, out, use_bn=head_use_bn, norm_last_layer=norm_last,
                nlayers=head_nlayers, hidden_dim=head_hidden_dim, bottleneck_dim=head_bottleneck_dim,
            )

        s_patch = _head(patch_out_dim, norm_last_layer) if use_patch else None
        t_patch = _head(patch_out_dim, True) if use_patch else None
        self.student = _StudentNet(student_backbone, _head(out_dim, norm_last_layer), s_patch)
        self.teacher = _TeacherNet(teacher_backbone, _head(out_dim, True), t_patch)

        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.out_dim = int(out_dim)
        self.patch_out_dim = int(patch_out_dim)

    @property
    def student_backbone(self) -> nn.Module:
        return self.student.backbone

    @property
    def teacher_backbone(self) -> nn.Module:
        return self.teacher.backbone

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        m = float(momentum)
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_((1.0 - m) * ps.detach().data)


def create_dino_full_ibot(
    *,
    arch: str = "tiny",
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int | None = None,
    depth: int | None = None,
    num_heads: int | None = None,
    out_dim: int = 4096,
    patch_out_dim: int = 4096,
    w_ibot: float = 0.1,
    head_hidden_dim: int = 2048,
    head_bottleneck_dim: int = 256,
    head_nlayers: int = 3,
    head_use_bn: bool = False,
    norm_last_layer: bool = True,
) -> DINOFulliBOTModule:
    overrides = {"embed_dim": embed_dim, "depth": depth, "num_heads": num_heads}
    sb = create_vit(arch, image_size=image_size, patch_size=patch_size, **overrides)
    tb = create_vit(arch, image_size=image_size, patch_size=patch_size, **overrides)
    return DINOFulliBOTModule(
        student_backbone=sb, teacher_backbone=tb, out_dim=out_dim, patch_out_dim=patch_out_dim,
        w_ibot=w_ibot, head_hidden_dim=head_hidden_dim, head_bottleneck_dim=head_bottleneck_dim,
        head_nlayers=head_nlayers, head_use_bn=head_use_bn, norm_last_layer=norm_last_layer,
    )
