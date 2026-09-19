"""masking.py -- block-wise mask sampling for MaskFeat, adapted from
BEiT's BEiTMaskGenerator (the sampler mmpretrain's MaskFeat config uses).
Samples rectangular blocks on the patch grid until ~mask_ratio of patches
are masked -- matching the official recipe's block-wise (not per-patch
Bernoulli) strategy."""

import torch


def block_mask(grid_size, mask_ratio=0.4, min_block=4, device="cpu"):
    """Returns a boolean mask (grid_size*grid_size,) with ~mask_ratio True."""
    num_patches = grid_size * grid_size
    num_masked_target = int(num_patches * mask_ratio)
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)

    attempts = 0
    while mask.sum().item() < num_masked_target and attempts < 30:
        remaining = num_masked_target - mask.sum().item()
        target_area = max(min_block, min(remaining, num_patches // 2))
        for _ in range(10):
            aspect = torch.empty(()).uniform_(0.3, 1 / 0.3).item()
            h = max(1, min(grid_size, int(round((target_area * aspect) ** 0.5))))
            w = max(1, min(grid_size, int(round((target_area / aspect) ** 0.5))))
            if h <= grid_size and w <= grid_size:
                break
        top = torch.randint(0, grid_size - h + 1, ()).item()
        left = torch.randint(0, grid_size - w + 1, ()).item()
        for i in range(top, top + h):
            for j in range(left, left + w):
                mask[i * grid_size + j] = True
        attempts += 1

    return mask


def batch_block_mask(batch_size, grid_size, mask_ratio=0.4, min_block=4, device="cpu"):
    return torch.stack([block_mask(grid_size, mask_ratio, min_block, device)
                        for _ in range(batch_size)])