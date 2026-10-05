"""Sanity-check line_saliency on REAL images with the project's REAL GaborBank,
and tell apart the two possible causes of a border artifact.
Run from the repo root (gabor.py and line_masking.py importable):

  python check_line_saliency.py --image_dir /home/pai-ng/Jamal/xpalm/smartphone_roi/6 --n 6

Reports ring-minus-interior saliency (outer ring of patches vs the rest) for
  RAW      : gabor_bank(images)                    (what the bank gives as-is)
  PADDED   : reflect-pad -> bank -> crop           (line_masking.padded_responses)
plus the image brightness of ring vs interior patches.
  RAW high, PADDED low   -> the bank's zero padding is the cause; use PADDED.
  PADDED still high      -> real image content at the border (dark/bright ROI edge).
Saves line_saliency_check.png (image | RAW | PADDED)."""
import argparse, glob, os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.getcwd())
from gabor import GaborBank, BASE_SCALE_LADDER
from line_masking import line_saliency, padded_responses

ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--img_size", type=int, default=112)
ap.add_argument("--num_patches", type=int, default=8)
ap.add_argument("--gabor_orient", type=int, default=8)
ap.add_argument("--gabor_num_scales", type=int, default=3)
ap.add_argument("--gabor_gamma", type=float, default=0.5)
ap.add_argument("--gabor_gray", type=int, default=0)       # SA-JEPA run uses --gabor_gray 0
ap.add_argument("--pad", type=int, default=16)
a = ap.parse_args()

files = sorted(sum([glob.glob(os.path.join(a.image_dir, e)) for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))[:a.n]
assert files, f"no images found in {a.image_dir}"
x01 = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((a.img_size,)*2), dtype=np.float32)/255.).permute(2,0,1)
                   for f in files])
x = (x01 - 0.5) / 0.5                                        # same Normalize([.5]*3, [.5]*3) as dataset.py
bank = GaborBank(n_orient=a.gabor_orient, scales=BASE_SCALE_LADDER[:a.gabor_num_scales],
                 gamma=a.gabor_gamma, per_channel=not bool(a.gabor_gray))
g = a.num_patches
with torch.no_grad():
    raw_resp = bank(x)
    print("gabor_bank output shape:", tuple(raw_resp.shape), "(must be 4-D: B, K, H, W)")
    s_raw = line_saliency(raw_resp, g)
    s_pad = line_saliency(padded_responses(bank, x, a.pad), g)
    bright = F.adaptive_avg_pool2d(x01.mean(1, keepdim=True), g).view(len(files), g, g)    # per-patch brightness in [0,1]

ring = torch.zeros(g, g, dtype=torch.bool); ring[0, :] = ring[-1, :] = True; ring[:, 0] = ring[:, -1] = True
def rmi(t):                                                  # ring minus interior, per image
    m = t.view(-1, g, g); return m[:, ring].mean(1) - m[:, ~ring].mean(1)
d_raw, d_pad = rmi(s_raw), rmi(s_pad)
d_br = bright[:, ring].mean(1) - bright[:, ~ring].mean(1)
print(f"\n{'image':<22}{'ring-int RAW':>14}{'ring-int PADDED':>17}{'brightness ring-int':>21}")
for i, f in enumerate(files):
    print(f"{os.path.basename(f):<22}{d_raw[i]:>+14.2f}{d_pad[i]:>+17.2f}{d_br[i]:>+21.2f}")
mr, mp, mb = d_raw.mean().item(), d_pad.mean().item(), d_br.mean().item()
print(f"\nmean: RAW {mr:+.2f} | PADDED {mp:+.2f} | brightness(ring-interior) {mb:+.2f}  (pixel scale 0..1)")
if mr < 0.3:
    print("-> OK: no border artifact in the raw responses.")
elif mp < 0.3:
    print("-> Zero-padding artifact CONFIRMED: PADDED removes it. Use padded_responses (patched main.py does).")
else:
    print("-> PADDED is still elevated: likely real border content (ring brightness differs from interior"
          " by %+.2f). Send me these numbers and the PNG; border patches need to be handled differently." % mb)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, len(files), figsize=(2.2*len(files), 6.8), squeeze=False)
    for i in range(len(files)):
        ax[0, i].imshow(x01[i].permute(1, 2, 0).numpy()); ax[1, i].imshow(s_raw[i].view(g, g).numpy(), cmap="magma")
        ax[2, i].imshow(s_pad[i].view(g, g).numpy(), cmap="magma")
        for r in range(3): ax[r, i].axis("off")
    for r, t in enumerate(["image", "RAW saliency", "PADDED saliency"]): ax[r, 0].set_title(t, fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig("line_saliency_check.png", dpi=120); print("saved line_saliency_check.png")
except Exception as e:
    print("(matplotlib unavailable; skipped figure:", e, ")")
