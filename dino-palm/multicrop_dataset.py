"""multicrop_dataset.py -- DataAugmentationDINO, adapted from official
(facebookresearch/dino/main_dino.py, fetched directly): 2 global crops +
N local crops. Crop PIXEL SIZES scaled proportionally from official's
224px-global/96px-local (ratio ~0.43) to this project's img_size, since
literally copying 224/96 makes no sense at 112px palmprint ROIs -- this
scaling is a necessary adaptation, not an independently verified number.
Augmentation composition itself (flip/jitter/grayscale/blur/solarize,
including the exact per-branch probabilities) IS verbatim from official.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from dataset import CASIADataset

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD = [0.5, 0.5, 0.5]


class _GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if np.random.rand() < self.p:
            return img.filter(ImageFilter.GaussianBlur(
                radius=np.random.uniform(self.radius_min, self.radius_max)))
        return img


class _Solarization:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        return img


class DataAugmentationDINO:
    def __init__(self, img_size, global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4), local_crops_number=8):
        local_size = max(16, int(round(img_size * 96 / 224)))

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_NORM_MEAN, _NORM_STD),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale,
                                        interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            _GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale,
                                        interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            _GaussianBlur(0.1),
            _Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale,
                                        interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            _GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class MultiCropDataset(CASIADataset):
    """Same samples/id_map contract as CASIADataset; __getitem__ returns
    a LIST of crops (2 global + N local) instead of one image."""

    def __init__(self, samples, id_map, img_size=112, aug_multiplier=1,
                 global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                 local_crops_number=8):
        super().__init__(samples, id_map, img_size, augment=True,
                         aug_multiplier=aug_multiplier)
        self.transform = DataAugmentationDINO(
            img_size, global_crops_scale, local_crops_scale, local_crops_number)

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        img = Image.open(s["path"]).convert("RGB")
        crops = self.transform(img)
        label = self.id_map[s["identity"]]
        return crops, label


def multicrop_collate(batch):
    """Batch is a list of (crops_list, label). Returns (list_of_crop_
    batches, labels) -- list[i] is a (B, 3, H, W) tensor for crop i,
    matching official's images[:2] / images slicing convention."""
    import torch
    crops_batch, labels = zip(*batch)
    n_crops = len(crops_batch[0])
    stacked = [torch.stack([sample[i] for sample in crops_batch]) for i in range(n_crops)]
    return stacked, torch.tensor(labels)
