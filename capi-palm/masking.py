"""masking.py -- ported VERBATIM from the official facebookresearch/capi
data.py (BlockMasking, InverseBlockMasking, collate_data_and_cast's masking
logic), confirmed against real source (previous version of this file was
reconstructed from the paper's prose alone and had two real bugs: n_predict
was NOT a fixed count but n_masked * prediction_subsampling, and indices
were computed per-image in a loop rather than batch-flattened as source
does). NumPy kept (matches source exactly) rather than ported to pure torch.
"""
import math
import random
import numpy as np
import torch


class BlockMasking:
    def __init__(self, input_size, roll=True, min_aspect=0.5, max_aspect=None):
        self.height, self.width = input_size
        self.roll = roll
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __call__(self, num_masking_patches=0):
        if num_masking_patches == 0:
            return np.zeros((self.height, self.width), dtype=bool)
        min_lar = max(self.log_aspect_ratio[0], np.log(num_masking_patches / (self.width ** 2)))
        max_lar = min(self.log_aspect_ratio[1], np.log(self.height ** 2 / (num_masking_patches + 1e-5)))
        aspect_ratio = math.exp(random.uniform(min_lar, max_lar))
        h = int(np.ceil(math.sqrt(num_masking_patches * aspect_ratio)))
        w = int(np.ceil(math.sqrt(num_masking_patches / aspect_ratio)))
        h = min(h, self.height)                     # guard: source assumes num_masking_patches
        w = min(w, self.width)                       # is always achievable within the grid
        top = random.randint(0, self.height - h)
        left = random.randint(0, self.width - w)
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[top:top + h, left:left + w] = True
        ids = np.where(mask.flatten())[0][:num_masking_patches]
        mask = np.zeros((self.height, self.width), dtype=bool).flatten()
        mask[ids] = True
        mask = mask.reshape((self.height, self.width))
        if self.roll:
            shift_x = random.randint(0, mask.shape[0] - 1)
            shift_y = random.randint(0, mask.shape[1] - 1)
            mask = np.roll(mask, (shift_x, shift_y), (0, 1))
        return mask


class InverseBlockMasking(BlockMasking):
    def __call__(self, num_masking_patches=0):
        mask = super().__call__(self.height * self.width - num_masking_patches)
        return ~mask


def collate_capi_masks(batch_size, grid, mask_ratio, prediction_subsampling, mask_roll=True, device="cpu"):
    """Ported from collate_data_and_cast's masking logic (image collation
    itself stays in this project's own CASIADataset/DataLoader -- only the
    mask/index construction is reproduced here). Returns (visible_indices,
    predict_indices), both flat LongTensors into the (batch_size*grid*grid)
    flattened token space, matching source's own indexing convention.

    n_predict per image = int(n_masked * prediction_subsampling), NOT a
    fixed count -- this was WRONG in an earlier version of this file, which
    hardcoded n_predict directly; --capi_n_predict has been REPLACED by
    --prediction_subsampling in config.py to match this correctly.
    """
    n_tokens = grid * grid
    n_masked = int(n_tokens * mask_ratio)
    n_predict = max(1, int(n_masked * prediction_subsampling))
    gen = InverseBlockMasking((grid, grid), roll=mask_roll)

    mask = torch.stack([torch.from_numpy(gen(n_masked)).flatten() for _ in range(batch_size)])   # (B, n_tokens) bool
    mask_indices_abs = mask.flatten().nonzero().reshape(batch_size, -1)                            # (B, n_masked)
    if mask_indices_abs.shape[1] < n_predict:
        raise SystemExit(f"n_predict={n_predict} > n_masked={mask_indices_abs.shape[1]} per image "
                          f"-- lower --prediction_subsampling or check --mask_ratio/--num_patches.")
    randperm = torch.argsort(torch.rand(batch_size, mask_indices_abs.shape[1]))[:, :n_predict]
    predict_indices_abs = torch.gather(mask_indices_abs, index=randperm, dim=1).flatten()

    visible_indices = (~mask).flatten().nonzero().flatten()
    return visible_indices.to(device), predict_indices_abs.to(device), n_predict
