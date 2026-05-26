"""Volume-token Transformer over slice embedding sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class VolumeTokenTransformer(nn.Module):
    """Prepend learnable volume token; TransformerEncoder on slices; read out token 0."""

    def __init__(
        self,
        dim: int,
        *,
        num_classes: int = 2,
        n_layers: int = 2,
        nhead: int = 8,
        dim_ff: int | None = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"embed_dim {dim} must be divisible by nhead {nhead}")
        dim_ff = dim_ff or 4 * dim
        self.vol_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, mask_valid: torch.Tensor) -> torch.Tensor:
        # x [B,T,D], mask_valid True = real slice
        b, _, d = x.shape
        vol = self.vol_token.expand(b, -1, -1)
        seq = torch.cat([vol, x], dim=1)
        kp = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~mask_valid], dim=1)
        h = self.encoder(seq, src_key_padding_mask=kp)
        return self.fc(h[:, 0])
