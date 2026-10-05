"""Compare normal (I-JEPA) vs line-guided target masking on ONE palm image.
Masked (target) patches are drawn as red squares; a third panel shows the saliency map.

Run from the repo root (needs models.py, gabor.py, line_masking.py importable):
  python visualize_masking.py --image /path/to/palm.jpg --seed 0
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.getcwd())
from models import patchify
from gabor import GaborBank, BASE_SCALE_LADDER
from line_masking import line_saliency, patchify_line_guided, target_topq_fraction

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="masking_comparison.png")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--img_size", type=int, default=112)
ap.add_argument("--num_patches", type=int, default=8)
ap.add_argument("--num_blocks", type=int, default=2)
ap.add_argument("--trg_ratio", type=float, nargs=2, default=[0.10, 0.15])
ap.add_argument("--ctx_ratio", type=float, nargs=2, default=[0.90, 1.00])
ap.add_argument("--eps", type=float, default=0.5)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--select", type=int, default=0, help="1 = multiply by orientation selectivity")
ap.add_argument("--gabor_orient", type=int, default=8)
ap.add_argument("--gabor_num_scales", type=int, default=3)
ap.add_argument("--gabor_gamma", type=float, default=0.5)
ap.add_argument("--gabor_gray", type=int, default=0)
a = ap.parse_args()

g = a.num_patches
ps = a.img_size // g

img = Image.open(a.image).convert("RGB").resize((a.img_size, a.img_size))
x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)[None]
x = (x - 0.5) / 0.5                                    # same Normalize as dataset.py

bank = GaborBank(n_orient=a.gabor_orient, scales=BASE_SCALE_LADDER[:a.gabor_num_scales],
                 gamma=a.gabor_gamma, per_channel=not bool(a.gabor_gray))
with torch.no_grad():
    saliency = line_saliency(bank(x), g, border=bank.pad, use_selectivity=bool(a.select))   # same as training

torch.manual_seed(a.seed)
_, tgt_normal = patchify(1, g, a.num_blocks, trg_ratio=tuple(a.trg_ratio), ctx_ratio=tuple(a.ctx_ratio))
torch.manual_seed(a.seed)
_, tgt_line = patchify_line_guided(saliency, 1, g, a.num_blocks, trg_ratio=tuple(a.trg_ratio),
                                   ctx_ratio=tuple(a.ctx_ratio), eps=a.eps, tau=a.tau)

def draw(ax, tgt_masks, title):
    ax.imshow(img, extent=(0, a.img_size, a.img_size, 0))
    idx = torch.cat([m[0] for m in tgt_masks]).unique().tolist()
    for i in idx:
        r, c = divmod(i, g)
        ax.add_patch(Rectangle((c * ps, r * ps), ps, ps, facecolor="red", alpha=0.25, linewidth=0))
        ax.add_patch(Rectangle((c * ps, r * ps), ps, ps, edgecolor="red", facecolor="none", linewidth=1.8))
    frac = target_topq_fraction(saliency, tgt_masks)
    ax.set_title(f"{title}\n{len(idx)} masked patches, {frac:.0%} in top-25% line saliency", fontsize=10)
    ax.axis("off")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
draw(axes[0], tgt_normal, "Normal masking (I-JEPA, uniform)")
draw(axes[1], tgt_line, f"Line-guided masking (eps={a.eps}, tau={a.tau})")
axes[2].imshow(img, extent=(0, a.img_size, a.img_size, 0))
axes[2].imshow(saliency[0].view(g, g).numpy(), cmap="magma", alpha=0.6, extent=(0, a.img_size, a.img_size, 0))
axes[2].set_title("Line saliency (bright = sampled more often)", fontsize=10); axes[2].axis("off")
plt.tight_layout()
plt.savefig(a.out, dpi=150, bbox_inches="tight", pad_inches=0.15)
print("saved", a.out)
