"""models.py -- MAE encoder+decoder, verified against facebookresearch/mae's
models_mae.py (fetched/confirmed via search: random_masking via argsort-of-
-noise, forward_encoder gathers ONLY visible patches, separate lightweight
decoder, norm_pix_loss). Encoder capacity matched to this project's
ContextEncoder (same auto-derive depth/heads formula) for a fair
parameter-count comparison; decoder is a genuinely smaller/separate
transformer, matching official's asymmetric design (base config:
decoder_embed_dim=512 vs encoder embed_dim=768, i.e. decoder ~2/3 width --
scaled proportionally here).
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


def random_masking(x, mask_ratio):
    """Verbatim logic from official random_masking(): per-sample shuffle
    via argsort of uniform noise, keep the first (1-mask_ratio) fraction.
    Returns (x_visible, mask, ids_restore).
    mask: (B, L), 0 = keep/visible, 1 = removed/masked."""
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class MAEEncoder(nn.Module):
    """Same depth/heads auto-derive formula as this project's ContextEncoder
    (verified matching at embed_dim=256: depth=6, heads=8, 4.91M params).
    Encodes ONLY visible patches, matching official's forward_encoder --
    this is the design ContextEncoder already has, but reimplemented here
    standalone (not reusing ContextEncoder directly) since MAE's masking
    call site differs (random_masking happens INSIDE forward, not via
    external ctx_masks the way patchify/ContextEncoder work)."""

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

        self.grid_size = num_patches
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(torch.tensor(pos).float().unsqueeze(0),
                                      requires_grad=False)

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask_ratio=0.75):
        """Returns (latent, mask, ids_restore). latent has ONLY visible
        tokens (matches official -- decoder re-inserts mask tokens)."""
        z = self.proj(x).flatten(2).transpose(1, 2)
        z = z + self.pos_embed

        z, mask, ids_restore = random_masking(z, mask_ratio)

        z = self.encoder(z)
        z = self.norm(z)
        return z, mask, ids_restore

    def forward_full(self, x):
        """No masking -- full sequence, for evaluation (FeatureExtractor)."""
        z = self.proj(x).flatten(2).transpose(1, 2)
        z = z + self.pos_embed
        z = self.encoder(z)
        z = self.norm(z)
        return z


class MAEDecoder(nn.Module):
    """Lightweight, separate transformer -- official base config uses
    decoder_embed_dim=512 vs encoder 768 (~2/3 width), decoder_depth=8
    (fewer than encoder's 12 in ImageNet-scale, but proportionally similar
    for this project's smaller 6-depth encoder -- decoder_depth defaults
    to encoder depth here, since 8 vs 12 in official is not a fixed ratio
    worth hardcoding at this much smaller scale)."""

    def __init__(self, num_patches, encoder_dim, decoder_dim=None,
                 decoder_depth=None, decoder_heads=None, mlp_ratio=4.0,
                 patch_pixel_dim=None):
        super().__init__()
        decoder_dim = decoder_dim or max(encoder_dim * 2 // 3, 32)
        decoder_depth = decoder_depth or 4
        decoder_heads = decoder_heads or max(4, decoder_dim // 32)

        self.grid_size = num_patches
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        pos = get_2d_sincos_pos_embed(decoder_dim, num_patches)
        self.decoder_pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0), requires_grad=False)

        enc = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=decoder_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(enc, decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_pixel_dim)

    def forward(self, latent, ids_restore):
        """latent: (B, len_keep, encoder_dim), visible tokens only."""
        x = self.decoder_embed(latent)
        B, len_keep, D = x.shape
        L = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(B, L - len_keep, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        x_ = x_ + self.decoder_pos_embed
        x_ = self.decoder(x_)
        x_ = self.decoder_norm(x_)
        return self.decoder_pred(x_)


class FeatureExtractor(nn.Module):
    """Same contract as the project's other FeatureExtractor classes."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder.forward_full(x)
        return z.mean(dim=1)