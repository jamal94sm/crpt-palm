"""multicrop_dataset.py -- DINO-style multi-crop augmentation for LeJEPA.
Paper's "Experiment Details 1": V=8 total views (Vg=2 global, Vl=6 local by
default here; paper's own inet1k default is Vl=6 giving V=8 -- see Table 1b,
row V=8 with Vg=2 is the paper's actual recommended operating point, not the
Vg=2,Vl=8,V=10 row sometimes quoted from the abbreviated recipe list).
Crop PIXEL sizes follow this project's patch grid convention (same reasoning
as dinov2-palm/multicrop_dataset.py): global = img_size, local = a smaller
exact multiple of the patch size. No masking is applied anywhere -- LeJEPA
views are plain augmented crops, never patch-masked.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
from PIL import Image, ImageFilter
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


class DataAugmentationLeJEPA:
    def __init__(self, global_size, local_size, global_scale, local_scale, n_local):
        def geometric(size, scale):
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=scale, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5)])
        color = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)])
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
        self.global_t = transforms.Compose([geometric(global_size, global_scale), color, _GaussianBlur(0.5), norm])
        self.local_t = transforms.Compose([geometric(local_size, local_scale), color, _GaussianBlur(0.5), norm])
        self.n_local = n_local

    def __call__(self, img):
        return [self.global_t(img), self.global_t(img)] + [self.local_t(img) for _ in range(self.n_local)]


class MultiCropDataset(CASIADataset):
    """Same samples/id_map/length contract as CASIADataset; __getitem__
    returns (list of 2 + n_local crops, label)."""

    def __init__(self, samples, id_map, img_size, aug_multiplier, global_scale, local_scale, n_local, local_size):
        super().__init__(samples, id_map, img_size, augment=True, aug_multiplier=aug_multiplier)
        self.transform = DataAugmentationLeJEPA(img_size, local_size, tuple(global_scale), tuple(local_scale), n_local)

    def __getitem__(self, idx):
        s = self.samples[idx % len(self.samples)]
        crops = self.transform(Image.open(s["path"]).convert("RGB"))
        return crops, self.id_map[s["identity"]]


def multicrop_collate(batch):
    """-> ([crop_0 (B,3,H,W), crop_1, local_0, ...], labels): view-major,
    matching sigreg_loss.lejepa_prediction_loss's expected (view, batch) layout."""
    crops_batch, labels = zip(*batch)
    stacked = [torch.stack([c[i] for c in crops_batch]) for i in range(len(crops_batch[0]))]
    return stacked, torch.tensor(labels)
