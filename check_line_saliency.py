"""Sanity-check line saliency on REAL images with the project's REAL GaborBank.
Run from the repo root (gabor.py and line_masking.py importable):

  python check_line_saliency.py --image_dir /home/pai-ng/Jamal/xpalm/smartphone_roi/6 --n 6

Compares three ways of computing saliency (ring-minus-interior, outer ring of
patches vs the rest; ~0 or negative = no border bias):
  RAW     : pool over all pixels of gabor_bank(images)          (zero-padding artifact)
  REFLECT : reflect-pad -> bank -> crop                          (can fold on vignetting)
  VALID   : ignore the bank.pad-wide border band when pooling    (what training uses)
Saves line_saliency_check.png (image | RAW | REFLECT | VALID)."""
import argparse, glob, os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.getcwd())
from gabor import GaborBank, BASE_SCALE_LADDER
from line_masking import line_saliency, padded_responses, ridge_saliency

ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--img_size", type=int, default=112)
ap.add_argument("--num_patches", type=int, default=8)
ap.add_argument("--gabor_orient", type=int, default=8)
ap.add_argument("--gabor_num_scales", type=int, default=3)
ap.add_argument("--gabor_gamma", type=float, default=0.5)
ap.add_argument("--gabor_gray", type=int, default=0)       # SA-JEPA run uses --gabor_gray 0
ap.add_argument("--select", type=int, default=0, help="1 = multiply by orientation selectivity")
ap.add_argument("--clip", type=float, default=2.0, help="clip saliency z-scores to [-clip, clip]; <=0 = off")
ap.add_argument("--ridge_sigmas", type=float, nargs="+", default=[1.0, 1.5, 2.0])
ap.add_argument("--ridge_line_frac", type=float, default=0.10)
a = ap.parse_args()

files = sorted(sum([glob.glob(os.path.join(a.image_dir, e)) for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))[:a.n]
assert files, f"no images found in {a.image_dir}"
x01 = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((a.img_size,)*2), dtype=np.float32)/255.).permute(2,0,1)
                   for f in files])
x = (x01 - 0.5) / 0.5                                        # same Normalize([.5]*3, [.5]*3) as dataset.py
bank = GaborBank(n_orient=a.gabor_orient, scales=BASE_SCALE_LADDER[:a.gabor_num_scales],
                 gamma=a.gabor_gamma, per_channel=not bool(a.gabor_gray))
g, sel = a.num_patches, bool(a.select)
with torch.no_grad():
    raw_resp = bank(x)
    print("gabor_bank output shape:", tuple(raw_resp.shape), f"| border band = bank.pad = {bank.pad}px")
    S = {"RAW":     line_saliency(raw_resp, g, use_selectivity=sel),
         "REFLECT": line_saliency(padded_responses(bank, x), g, use_selectivity=sel),
         "VALID":   line_saliency(raw_resp, g, border=bank.pad, use_selectivity=sel, clip=a.clip if a.clip > 0 else None)}
    s_ridge, ridge_lines = ridge_saliency(x, g, sigmas=tuple(a.ridge_sigmas), line_frac=a.ridge_line_frac,
                                          clip=a.clip if a.clip > 0 else None, return_lines=True)
    S["RIDGE"] = s_ridge
    bright = F.adaptive_avg_pool2d(x01.mean(1, keepdim=True), g).view(len(files), g, g)

ring = torch.zeros(g, g, dtype=torch.bool); ring[0, :] = ring[-1, :] = True; ring[:, 0] = ring[:, -1] = True
def rmi(t):
    m = t.view(-1, g, g); return m[:, ring].mean(1) - m[:, ~ring].mean(1)
D = {k: rmi(v) for k, v in S.items()}
d_br = bright[:, ring].mean(1) - bright[:, ~ring].mean(1)
print(f"\n{'image':<22}{'RAW':>8}{'REFLECT':>10}{'VALID':>8}{'RIDGE':>8}{'brightness':>12}   (ring - interior)")
for i, f in enumerate(files):
    print(f"{os.path.basename(f):<22}{D['RAW'][i]:>+8.2f}{D['REFLECT'][i]:>+10.2f}{D['VALID'][i]:>+8.2f}{D['RIDGE'][i]:>+8.2f}{d_br[i]:>+12.2f}")
mv = D["VALID"].mean().item()
print(f"\nmean: " + " | ".join(f"{k} {v.mean().item():+.2f}" for k, v in D.items()) + f" | brightness {d_br.mean().item():+.2f}")
if mv < 0.3:
    print("-> OK: VALID has no border bias. Now check the PNG: bright VALID cells should follow the principal lines.")
else:
    print("-> VALID is still elevated: the border itself has line-like content (fingers, ROI edge, strong shading).\n"
          "   Send me these numbers and the PNG.")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    titles = ["image", "Gabor saliency (VALID)", "Ridge: detected line pixels", "Ridge saliency (line density)"]
    fig, ax = plt.subplots(4, len(files), figsize=(2.2*len(files), 9), squeeze=False)
    ext = (0, a.img_size, a.img_size, 0)
    for i in range(len(files)):
        im = x01[i].permute(1, 2, 0).numpy()
        ax[0, i].imshow(im)
        ax[1, i].imshow(im, extent=ext); ax[1, i].imshow(S["VALID"][i].view(g, g).numpy(), cmap="magma", alpha=0.65, extent=ext)
        ov = np.zeros((a.img_size, a.img_size, 4)); ov[ridge_lines[i, 0].numpy()] = [1, 0, 0, 1]
        ax[2, i].imshow(im, extent=ext); ax[2, i].imshow(ov, extent=ext)
        ax[3, i].imshow(im, extent=ext); ax[3, i].imshow(S["RIDGE"][i].view(g, g).numpy(), cmap="magma", alpha=0.65, extent=ext)
        for r in range(4): ax[r, i].axis("off")
    for r, t in enumerate(titles): ax[r, 0].set_title(t, fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig("line_saliency_check.png", dpi=120); print("saved line_saliency_check.png")
except Exception as e:
    print("(matplotlib unavailable; skipped figure:", e, ")")
