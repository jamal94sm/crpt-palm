"""models.py -- DINO's student/teacher ViT + projection head, verified
against facebookresearch/dino's vision_transformer.py/main_dino.py
(fetched directly). Encoder capacity matched to this project's
ContextEncoder convention; DINOHead's out_dim scaled down from official's
65536 (tuned for ImageNet-scale class diversity) proportionally to
embed_dim, same treatment as BYOL/Barlow Twins' oversized projectors.

CLS token IS used here (unlike this project's other baselines) because
DINO's centering+sharpening mechanism is designed around a single
distilled token's output distribution, not a patch-level objective --
this is a disclosed, necessary deviation from this project's mean-pool
convention, matching official's own design intent for this specific
method.
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


def _safe_heads(dim, target_ratio=32):
    """Guarantees dim % heads == 0 -- lesson learned from mae-palm's
    divisibility bugs. Walks down from the target ratio to find an
    exact divisor, never asserts on a non-divisor combination."""
    heads = max(1, dim // target_ratio)
    while dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


class DinoViT(nn.Module):
    """CLS-token ViT. Same depth auto-derive formula as ContextEncoder,
    but heads use _safe_heads() to guarantee a valid combination at any
    embed_dim (ContextEncoder's own formula only happens to work at the
    specific embed_dim values used elsewhere in this project)."""

    def __init__(self, image_size, num_patches, embed_dim,
                 depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        patch_h = H // num_patches
        patch_w = W // num_patches

        if num_heads is None:
            num_heads = _safe_heads(embed_dim, 32)
        if depth is None:
            depth = min(6, embed_dim // 64 + 2)

        self.grid_size = num_patches
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        pos = torch.tensor(pos).float().unsqueeze(0)
        cls_pos = torch.zeros(1, 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.cat([cls_pos, pos], dim=1),
                                      requires_grad=False)

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def _interpolate_pos_embed(self, n_patches, device):
        """Official DINO supports variable input resolution (multi-crop's
        differently-sized local vs. global crops) via interpolating the
        position embedding to match the actual patch grid at each forward
        call. self.pos_embed was built for self.grid_size**2 + 1
        positions; here we bicubically resize its patch portion to
        whatever grid the current input actually produces."""
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]

        if n_patches == self.grid_size * self.grid_size:
            return self.pos_embed

        new_grid = int(round(n_patches ** 0.5))
        dim = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, self.grid_size, self.grid_size, dim).permute(0, 3, 1, 2)
        patch_pos = torch.nn.functional.interpolate(
            patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        """Returns CLS token output only, (B, embed_dim) -- matches
        official's backbone-then-head chaining. Supports variable input
        size (multi-crop) via interpolated position embedding."""
        B = x.size(0)
        z = self.proj(x).flatten(2).transpose(1, 2)
        pos = self._interpolate_pos_embed(z.size(1), x.device)
        cls = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1) + pos
        z = self.encoder(z)
        z = self.norm(z)
        return z[:, 0]

    def forward_patches(self, x):
        """Full sequence including CLS, for a mean-pool eval option."""
        B = x.size(0)
        z = self.proj(x).flatten(2).transpose(1, 2)
        pos = self._interpolate_pos_embed(z.size(1), x.device)
        cls = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1) + pos
        z = self.encoder(z)
        z = self.norm(z)
        return z


class DINOHead(nn.Module):
    """3-layer MLP + weight-normalized final linear layer, matching
    official's DINOHead structure (nlayers=3, bottleneck_dim, optional
    norm_last_layer). out_dim scaled down from official's 65536 (tuned
    for ImageNet's huge class diversity) proportionally to embed_dim --
    same treatment as BYOL/Barlow Twins' oversized official projectors."""

    def __init__(self, in_dim, out_dim, hidden_dim=None, bottleneck_dim=None,
                 use_bn=False, norm_last_layer=True):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        bottleneck_dim = bottleneck_dim or max(in_dim // 4, 16)

        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class FeatureExtractor(nn.Module):
    """Eval-time feature extractor. use_cls=True (default) matches
    official's own CLS-token convention for this method; use_cls=False
    mean-pools instead, for consistency with this project's other
    baselines if preferred."""

    def __init__(self, encoder, use_cls=True):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.use_cls = use_cls

    def forward(self, x):
        with torch.no_grad():
            if self.use_cls:
                return self.encoder(x)
            z = self.encoder.forward_patches(x)
            return z[:, 1:].mean(dim=1)   # exclude CLS, mean-pool patches
