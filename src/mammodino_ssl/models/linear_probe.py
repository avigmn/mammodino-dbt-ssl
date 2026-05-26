"""Frozen TinyViT encoder + linear classifier (downstream probe)."""

from __future__ import annotations

import torch
from torch import nn

from mammodino_ssl.models.toy_vit import TinyViT


class FrozenTinyViTLinearProbe(nn.Module):
    """Train only a small head on top of frozen CLS features."""

    def __init__(
        self,
        backbone: TinyViT,
        *,
        num_classes: int = 10,
        head_type: str = "linear",
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = backbone
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        in_dim = int(backbone.embed_dim)
        ht = str(head_type).lower()
        if ht == "linear":
            self.head = nn.Linear(in_dim, num_classes)
        elif ht == "mlp":
            h = int(mlp_hidden_dim) if mlp_hidden_dim is not None else max(128, in_dim)
            self.head = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, num_classes),
            )
        else:
            raise ValueError(f"unknown head_type={head_type!r} (expected 'linear' or 'mlp')")

    def train(self, mode: bool = True) -> FrozenTinyViTLinearProbe:
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        with torch.no_grad():
            cls, _ = self.encoder(x)
        return self.head(cls)
