"""masking.py -- iBOT block masking, as in dinov2 (data/masking.py + data/collate.py).

MaskingGenerator: __init__ and __call__ are identical to dinov2's (read directly); _mask() is
the BEiT block-sampling routine dinov2 inherits (same class signature as microsoft/unilm beit).
build_ibot_masks(): the mask part of collate_data_and_cast (read directly), run here on the
main process once per step instead of inside DataLoader workers.
"""
import math
import random
import numpy as np
import torch


class MaskingGenerator:
    def __init__(self, input_size, num_masking_patches=None, min_num_patches=4,
                 max_num_patches=None, min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top: top + h, left: left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:      # only accept blocks that add patches
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask


def make_mask_generator(grid):
    """Official: max_num_patches = 0.5 * n_tokens, min_num_patches = 4 (on a 16x16 grid).
    min_num_patches is capped on tiny grids so a 4-patch minimum cannot exceed the request."""
    n_tokens = grid * grid
    return MaskingGenerator(input_size=(grid, grid), min_num_patches=min(4, max(1, n_tokens // 16)),
                            max_num_patches=0.5 * n_tokens)


def build_ibot_masks(n_global_total, n_tokens, mask_ratio_tuple, mask_probability, mask_generator):
    """Returns (masks (Bg,N) bool, mask_indices (n_masked,) into masks.flatten(),
    masks_weight (n_masked,)). Bg = 2 * batch (both global crops, crop-major order).
    A `mask_probability` fraction of the global crops gets a block mask whose ratio is stratified
    over [ratio_min, ratio_max]; the rest get none; order is shuffled (official)."""
    B, N = n_global_total, n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    masks_list = []
    for i in range(n_samples_masked):
        prob_min, prob_max = float(probs[i]), float(probs[i + 1])
        masks_list.append(torch.from_numpy(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.from_numpy(mask_generator(0)))
    random.shuffle(masks_list)
    masks = torch.stack(masks_list).flatten(1)
    mask_indices = masks.flatten().nonzero().flatten()
    masks_weight = (1 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, mask_indices, masks_weight
