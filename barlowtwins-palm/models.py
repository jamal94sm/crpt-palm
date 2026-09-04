"""models.py -- VICReg encoder + expander + eval-time feature extractor.

Encoder mirrors crpt-palm's ContextEncoder/TargetEncoder trunk exactly
(patch conv, 2D sincos pos-embed, pre-norm TransformerEncoder, final
LayerNorm, depth/heads auto-derived from embed_dim by the same formula) so
encoder capacity is matched 1:1 with the proposed method. The only
differences: VICReg's encoder always sees the FULL image (no masking, no
mask-token machinery) and is fully trainable end-to-end (no EMA target --
VICReg doesn't use one).
"""

import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class Encoder(nn.Module):
    """Full-image ViT encoder. Same recipe as crpt-palm's ContextEncoder /
    TargetEncoder, minus masking, and WITH gradients enabled throughout."""

    def __init__(self, image_size, num_patches, embed_dim,
                 depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        patch_h = H // num_patches
        patch_w = W // num_patches

        if num_heads is None:
            num_heads = max(4, embed_dim // 32)
        if depth is None:
            depth = min(6, embed_dim // 64 + 2)

        self.proj = nn.Conv2d(3, embed_dim,
                               kernel_size=(patch_h, patch_w),
                               stride=(patch_h, patch_w))
        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0), requires_grad=False)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        z = self.proj(x).flatten(2).transpose(1, 2)
        z = z + self.pos_embed
        z = self.encoder(z)
        z = self.norm(z)
        return z


class Expander(nn.Module):
    """3-layer MLP projector, as in the VICReg paper."""

    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FeatureExtractor(nn.Module):
    """Wraps the encoder for eval: mean-pool over patch tokens.
    forward(x) -> [B, embed_dim] -- same contract evaluate.py's
    extract_features / run_full_eval expects (matches crpt-palm's
    CompNetBackbone / FeatModule contract). No freezing here: encoder.train()
    / encoder.eval() is toggled by the training loop; calling
    feature_extractor.eval() (which evaluate.py already does) recursively
    sets the wrapped encoder to eval mode too."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        z = self.encoder(x)
        return z.mean(dim=1)
