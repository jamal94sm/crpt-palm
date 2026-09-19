"""patchify.py -- pixel target construction, verbatim logic from official
patchify()/forward_loss() (norm_pix_loss confirmed via mae_st/PRETRAIN.md:
"Here we use --norm_pix_loss as the target for better representation
learning" -- official's own recommended, non-default-off setting)."""

import torch


def patchify(imgs, patch_size):
    """imgs: (B, 3, H, W) -> (B, num_patches, patch_size*patch_size*3)."""
    B, C, H, W = imgs.shape
    h = H // patch_size
    w = W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = torch.einsum('bchpwq->bhwpqc', x)
    x = x.reshape(B, h * w, patch_size * patch_size * C)
    return x


def normalize_pixel_target(target, eps=1e-6):
    """norm_pix_loss: per-patch mean/var normalization of the target
    before computing MSE -- official's recommended setting."""
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    return (target - mean) / (var + eps) ** 0.5