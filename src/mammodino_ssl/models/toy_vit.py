"""Small ViT backbone for toy SSL (torch.nn only)."""

from __future__ import annotations

import torch
from torch import nn


class _MHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.softmax(dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(y)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHA(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """CLS + patch tokens. Default 224 / 16 → 14×14 patches."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        num_patches = self.grid * self.grid
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _encode_tokens(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes patch tokens (B,N,D) into CLS + patch outputs."""
        b = patch_tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1) + self.pos_embed
        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns CLS (B, D) and patch tokens (B, N, D)."""
        b = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        if patches.shape[0] != b:
            raise ValueError("patch batch mismatch")
        return self._encode_tokens(patches)

    def forward_masked(self, x: torch.Tensor, patch_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with patch-level masking before transformer blocks.
        patch_mask shape: (B, N), True where patch is masked.
        """
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B,N,D)
        if patch_mask.ndim != 2 or patch_mask.shape[:2] != patches.shape[:2]:
            raise ValueError(
                f"patch_mask must be (B,N) matching tokenized patches; "
                f"got {tuple(patch_mask.shape)} vs patches {tuple(patches.shape)}"
            )
        m = patch_mask.bool().unsqueeze(-1)
        mask_tok = self.mask_token.expand(patches.size(0), patches.size(1), -1)
        patches = torch.where(m, mask_tok, patches)
        return self._encode_tokens(patches)


def create_tiny_vit(**kwargs: object) -> TinyViT:
    return TinyViT(**kwargs)
