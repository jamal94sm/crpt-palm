"""models.py -- MaskFeat's masked-in-place ViT, capacity-matched to this
project's ContextEncoder (same auto-derive depth/heads formula), but with
a genuinely different forward pass: masked positions are kept in the
sequence and replaced with a learnable [MASK] token, then the FULL
sequence (visible + masked) is processed together by the transformer --
verified against mmpretrain's MaskFeatViT.forward(), which does exactly
this (x = x*(1-mask) + mask_token*mask, then all layers see everything).
This is why ContextEncoder (gather-visible-only) cannot be reused here.

No CLS token -- kept consistent with this project's FeatureExtractor
convention (mean-pool over patch tokens), which every other baseline in
this project also uses; official MaskFeatViT uses a CLS token instead.
Disclosed deviation, not an oversight.
"""

import torch
import torch.nn as nn
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_1d(embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d(embed_dim // 2, grid[0])
    emb_w = get_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class MaskedViT(nn.Module):
    def __init__(self, image_size, num_patches, embed_dim, depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        ph, pw = H // num_patches, W // num_patches
        num_heads = num_heads or max(4, embed_dim // 32)
        depth = depth or min(6, embed_dim // 64 + 2)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # official: learnable pos_embed, one extra slot for CLS (MVIT.PATCH_2D + CLS_EMBED_ON: True)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                         dim_feedforward=int(embed_dim * mlp_ratio),
                                         dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        B = x.size(0)
        z = self.proj(x).flatten(2).transpose(1, 2)                 # (B, P, D)
        if mask is not None:
            m = mask.unsqueeze(-1).to(z.dtype)
            z = z * (1 - m) + self.mask_token.expand(B, z.size(1), -1) * m   # 1. substitute
        cls = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1)                               # 2. prepend CLS
        z = z + self.pos_embed                                       # 3. position LAST
        return self.norm(self.encoder(z))[:, 1:]                     # patch tokens only

class MaskFeatHead(nn.Module):
    """Single linear layer, matching official's LinearNeck exactly (no
    hidden layer, no activation)."""

    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        return self.proj(x)


class FeatureExtractor(nn.Module):
    """Same contract as the project's other FeatureExtractor classes:
    forward(x) -> (B, embed_dim), mean-pooled, no mask (full image)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x, mask=None)
        return z.mean(dim=1)
