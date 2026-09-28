"""multicrop_dataset.py -- DINO/DINOv2 multi-crop augmentation on the project's CASIADataset.

Official DINOv2 crop *scales* (global 0.32-1.0, local 0.05-0.32) and per-crop photometric
recipe (flip, colour-jitter p=0.8, grayscale p=0.2, blur p=1.0/0.1/0.5, solarise p=0.2 on
global crop 2). Crop *pixel sizes* follow the patch grid: global = img_size, local =
local_grid*patch_px with local_grid = round(6/16 * grid) -- the same local/global token ratio
(36/256) as the official 96px/224px setting, but an exact multiple of the patch size.
Normalisation is the project's Normalize(0.5, 0.5) so eval-time features are comparable.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from dataset import CASIADataset

_MEAN, _STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


class _GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p, self.rmin, self.rmax = p, radius_min, radius_max

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.rmin, self.rmax)))
        return img


class _Solarize:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img, threshold=128) if random.random() < self.p else img


class DataAugmentationDINOv2:
    def __init__(self, global_size, local_size, global_scale, local_scale, n_local):
        def geometric(size, scale):
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5)])
        color = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)])
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
        self.g1 = transforms.Compose([geometric(global_size, global_scale), color, _GaussianBlur(1.0), norm])
        self.g2 = transforms.Compose([geometric(global_size, global_scale), color, _GaussianBlur(0.1),
                                      _Solarize(0.2), norm])
        self.local = transforms.Compose([geometric(local_size, local_scale), color, _GaussianBlur(0.5), norm])
        self.n_local = n_local

    def __call__(self, img):
        return [self.g1(img), self.g2(img)] + [self.local(img) for _ in range(self.n_local)]


class MultiCropDataset(CASIADataset):
    """Same samples / id_map / length contract as CASIADataset; __getitem__ returns
    (list of 2 + n_local crops, label)."""

    def __init__(self, samples, id_map, img_size, aug_multiplier, global_scale, local_scale,
                 n_local, local_size):
        super().__init__(samples, id_map, img_size, augment=True, aug_multiplier=aug_multiplier)
        self.transform = DataAugmentationDINOv2(img_size, local_size, tuple(global_scale),
                                                tuple(local_scale), n_local)

    def __getitem__(self, idx):
        s = self.samples[idx % len(self.samples)]
        crops = self.transform(Image.open(s["path"]).convert("RGB"))
        return crops, self.id_map[s["identity"]]


def multicrop_collate(batch):
    """-> ([crop_0 (B,3,H,W), crop_1, local_0 (B,3,h,w), ...], labels): crop-major, as in
    dinov2's collate_data_and_cast (all samples' crop i are stacked together)."""
    crops_batch, labels = zip(*batch)
    stacked = [torch.stack([c[i] for c in crops_batch]) for i in range(len(crops_batch[0]))]
    return stacked, torch.tensor(labels)
