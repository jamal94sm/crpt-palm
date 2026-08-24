"""paired_dataset.py -- two-view dataset for VICReg, following the ORIGINAL
VICReg two-branch augmentation recipe (asymmetric blur/solarization across
the two views, as in facebookresearch/vicreg's augmentations.py), adapted
only where the domain requires it:
  - crop size = img_size (the original hardcodes 224 for ImageNet)
  - crop scale = (0.7, 1.0) instead of the original's default (0.08, 1.0) --
    an 8%-area crop destroys a 112x112 palmprint ROI; this matches the
    crop strength CASIADataset's own augment=True branch already uses.
  - normalization = [0.5]*3 / [0.5]*3, matching CASIADataset's eval-time
    normalization elsewhere in this pipeline (not ImageNet stats).

Per-epoch TRAINING SIZE still matches the proposed method exactly: __len__
is unchanged from CASIADataset, i.e. len(samples) * aug_multiplier (same
--aug_multiplier flag, same value, same config.py). Same number of dataset
items per epoch -> same number of batches -> same number of optimizer steps
per epoch. Each VICReg item just carries two views instead of one.

dataset.py is untouched (a straight copy of crpt-palm's); this file only
adds a new Dataset class on top of it.
"""

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

from dataset import CASIADataset


class _GaussianBlur:
    """PIL Gaussian blur, applied with probability p.
    Matches facebookresearch/vicreg's augmentations.py GaussianBlur."""
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class _Solarization:
    """PIL solarize, applied with probability p.
    Matches facebookresearch/vicreg's augmentations.py Solarization."""
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        return img


def _build_view_transform(img_size, blur_p, solarize_p):
    """One branch of the original VICReg TrainTransform (crop/flip/jitter/
    grayscale identical structure to the official recipe; crop size, crop
    scale, and normalization adapted for this palmprint pipeline -- see
    module docstring)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        _GaussianBlur(p=blur_p),
        _Solarization(p=solarize_p),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


class PairedCASIADataset(CASIADataset):
    """Same samples / id_map / __len__ contract as CASIADataset, so
    per-epoch training size still equals len(samples) * aug_multiplier --
    identical to the proposed method's train_loader. __getitem__ returns
    (view1, view2, label) using the ORIGINAL VICReg asymmetric recipe:
    view1 is always blurred, never solarized; view2 is rarely blurred,
    sometimes solarized. Crop/flip/jitter/grayscale are drawn independently
    per branch but from the same distribution on both branches."""

    def __init__(self, samples, id_map, img_size=112, augment=True,
                 aug_multiplier=1):
        super().__init__(samples, id_map, img_size, augment=augment,
                          aug_multiplier=aug_multiplier)
        # Overwrite the single-view transform CASIADataset.__init__ built --
        # VICReg uses its own two-branch recipe instead.
        self.transform = _build_view_transform(img_size, blur_p=1.0, solarize_p=0.0)
        self.transform_prime = _build_view_transform(img_size, blur_p=0.1, solarize_p=0.2)

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        img = Image.open(s["path"]).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform_prime(img)
        label = self.id_map[s["identity"]]
        return view1, view2, label
