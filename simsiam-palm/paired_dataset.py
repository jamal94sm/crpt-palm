"""paired_dataset.py -- two-view dataset for SimSiam (Chen & He, CVPR 2021).

SimSiam's official augmentation (matching the paper / facebookresearch/simsiam)
is SYMMETRIC: the SAME transform pipeline is used for both views -- unlike
VICReg's asymmetric blur/solarize split. No solarization at all here.

Crop scale (0.7, 1.0) instead of the paper's ImageNet default (0.2, 1.0) --
same palmprint-specific adaptation used elsewhere in this project (an
8%-area crop destroys a 112x112 palmprint ROI's ridge structure).

dataset.py is untouched (shared, copied verbatim); this file only adds a
dataset class on top of it.
"""

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from dataset import CASIADataset

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD = [0.5, 0.5, 0.5]


class _GaussianBlur:
    """PIL Gaussian blur, applied with probability p. Sigma range matches
    SimSiam's official recipe (same as MoCo v2 / SimCLR)."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.uniform(0.1, 2.0)
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


def _build_view_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        _GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])


class PairedCASIADataset(CASIADataset):
    """Same samples/id_map/__len__ contract as CASIADataset. __getitem__
    returns (view1, view2, label) -- both from the SAME transform
    pipeline (symmetric), two independent random draws."""

    def __init__(self, samples, id_map, img_size=112, augment=True,
                 aug_multiplier=1):
        super().__init__(samples, id_map, img_size, augment=augment,
                          aug_multiplier=aug_multiplier)
        self.transform = _build_view_transform(img_size)

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        img = Image.open(s["path"]).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        label = self.id_map[s["identity"]]
        return view1, view2, label
