"""paired_dataset.py -- two-view wrapper around CASIADataset for VICReg's
invariance term.

VICReg needs two independently-augmented views of the SAME image to compute
its invariance loss. To keep the comparison with the proposed method fair,
both views are drawn from the EXACT SAME transform pipeline CASIADataset
already builds for the shared --aug_multiplier dataset enlargement (Resize
-> RandomResizedCrop -> RandomHorizontalFlip -> ColorJitter -> GaussianBlur
-> RandomRotation -> ToTensor -> Normalize). No extra or different
augmentation is introduced for VICReg anywhere in this file.

dataset.py itself is untouched -- a straight copy of crpt-palm's. This file
only adds a thin subclass on top of it.
"""

from PIL import Image

from dataset import CASIADataset


class PairedCASIADataset(CASIADataset):
    """Same samples / id_map / transform as CASIADataset; __getitem__ returns
    (view1, view2, label) -- two independent draws through self.transform on
    the same loaded image, instead of CASIADataset's single draw."""

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        img = Image.open(s["path"]).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        label = self.id_map[s["identity"]]
        return view1, view2, label