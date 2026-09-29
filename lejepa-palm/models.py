"""models.py -- LeJEPA uses a SINGLE encoder, no predictor, no target/EMA
network (paper: "removal of the predictor and teacher-student architecture
without suffering from collapse", Section 6.1 "Removal of popular heuristics").

ContextEncoder is copied verbatim from this project's models.py (same class,
same forward(x, masks) signature) but is used here WITHOUT any masking -- LeJEPA
operates on full, unmasked augmented crops (multi-crop, DINO-style), not on
JEPA-style context/target patch splits. patchify()/apply_masks() from the
JEPA family are NOT used anywhere in this baseline.
"""
import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float); omega /= embed_dim / 2.; omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1); out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    gh, gw = np.arange(grid_size, dtype=float), np.arange(grid_size, dtype=float)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape([2, 1, grid_size, grid_size])
    return np.concatenate([get_1d_sincos_pos_embed(embed_dim // 2, grid[0]),
                           get_1d_sincos_pos_embed(embed_dim // 2, grid[1])], axis=1)


def _safe_heads(dim, ratio=32):
    h = max(1, dim // ratio)
    while dim % h != 0 and h > 1:
        h -= 1
    return h


class LeJepaEncoder(nn.Module):
    """Identical structure/capacity convention to this project's ContextEncoder
    (same depth/heads rule, 4x MLP, pre-norm, fixed 2-D sin-cos pos-embed),
    but ALWAYS runs full-image forward (no context/target masking -- there is
    no masking concept in LeJEPA at all). Mean-pools patch tokens for the
    embedding fed to the LeJEPA loss (same pooling convention as this
    project's FeatureExtractor, so train-time and eval-time features match)."""

    def __init__(self, image_size, num_patches, embed_dim, depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        ph, pw = H // num_patches, W // num_patches
        num_heads = num_heads or _safe_heads(embed_dim)
        depth = depth or min(6, embed_dim // 64 + 2)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=(ph, pw), stride=(ph, pw))
        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(torch.tensor(pos).float().unsqueeze(0), requires_grad=False)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                         dim_feedforward=int(embed_dim * mlp_ratio),
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def _pos(self, n_patches):
        """Position embedding for a crop with n_patches tokens. Global crops
        match self.pos_embed's native grid exactly; local crops (smaller
        pixel size -> fewer patches) get a bicubic-interpolated version --
        same fix as this project's dino-palm/dinov2-palm baselines apply
        for their own local-crop / masked-grid mismatches."""
        native = self.pos_embed.shape[1]
        if n_patches == native:
            return self.pos_embed
        g_native = int(round(native ** 0.5))
        g_new = int(round(n_patches ** 0.5))
        assert g_new * g_new == n_patches, f"non-square token grid ({n_patches} tokens)"
        pos = self.pos_embed.reshape(1, g_native, g_native, -1).permute(0, 3, 1, 2)
        pos = torch.nn.functional.interpolate(pos, size=(g_new, g_new), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, n_patches, -1)

    def forward(self, x):
        """x: (B,3,H,W) -> (B, embed_dim), mean-pooled patch embedding."""
        z = self.proj(x).flatten(2).transpose(1, 2)
        z = z + self._pos(z.shape[1])
        z = self.norm(self.encoder(z))
        return z.mean(dim=1)


class FeatureExtractor(nn.Module):
    """forward(x) -> (B, embed_dim). LeJEPA has no target encoder, so this
    wraps the SAME encoder used for training -- unlike JEPA-family baselines
    which evaluate the (separately EMA-updated) target encoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)
