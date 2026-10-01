"""Sanity-check line_saliency on REAL images with the project's REAL GaborBank.
Run from the repo root (where gabor.py and line_masking.py are importable):

  python check_line_saliency.py --image_dir /home/pai-ng/Jamal/xpalm/smartphone_roi/6 --n 6

Prints: gabor_bank output shape (must be 4-D), saliency stats, and whether the
OUTER RING of patches scores higher than the interior (a border artifact of the
filter padding, which would make guided masking favour image edges). Saves
line_saliency_check.png (image | 8x8 saliency) when matplotlib is available."""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.getcwd())
from gabor import GaborBank, BASE_SCALE_LADDER
from line_masking import line_saliency

ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--img_size", type=int, default=112)
ap.add_argument("--num_patches", type=int, default=8)
ap.add_argument("--gabor_orient", type=int, default=8)
ap.add_argument("--gabor_num_scales", type=int, default=3)
ap.add_argument("--gabor_gamma", type=float, default=0.5)
ap.add_argument("--gabor_gray", type=int, default=0)       # SA-JEPA run uses --gabor_gray 0
a = ap.parse_args()

files = sorted(sum([glob.glob(os.path.join(a.image_dir, e)) for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))[:a.n]
assert files, f"no images found in {a.image_dir}"
x = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((a.img_size,)*2), dtype=np.float32)/255.).permute(2,0,1)
                 for f in files])
x = (x - 0.5) / 0.5                                          # same Normalize([.5]*3, [.5]*3) as dataset.py
bank = GaborBank(n_orient=a.gabor_orient, scales=BASE_SCALE_LADDER[:a.gabor_num_scales],
                 gamma=a.gabor_gamma, per_channel=not bool(a.gabor_gray))
with torch.no_grad():
    resp = bank(x)
    print("gabor_bank output shape:", tuple(resp.shape), "(must be 4-D: B, K, H, W)")
    s = line_saliency(resp, a.num_patches)
g = a.num_patches
ring = torch.zeros(g, g, dtype=torch.bool); ring[0, :] = ring[-1, :] = True; ring[:, 0] = ring[:, -1] = True
for i, f in enumerate(files):
    m = s[i].view(g, g)
    print(f"{os.path.basename(f)}: min={m.min():.2f} max={m.max():.2f} | outer-ring mean={m[ring].mean():+.2f} interior mean={m[~ring].mean():+.2f}")
d = (s.view(-1, g, g)[:, ring].mean() - s.view(-1, g, g)[:, ~ring].mean()).item()
print(f"\nring - interior (avg over images): {d:+.2f}  ->",
      "OK" if d < 0.3 else "WARNING: border patches look artificially salient (check GaborBank padding; reflect/replicate padding avoids this)")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(files), figsize=(2.2*len(files), 4.6), squeeze=False)
    for i in range(len(files)):
        ax[0, i].imshow(((x[i].permute(1, 2, 0)*0.5+0.5).clamp(0, 1)).numpy()); ax[0, i].axis("off")
        ax[1, i].imshow(s[i].view(g, g).numpy(), cmap="magma"); ax[1, i].axis("off")
    plt.tight_layout(); plt.savefig("line_saliency_check.png", dpi=120); print("saved line_saliency_check.png")
except Exception as e:
    print("(matplotlib unavailable; skipped figure:", e, ")")
