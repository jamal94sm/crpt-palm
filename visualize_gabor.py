"""visualize_gabor.py -- HOG-style visualisation of the Gabor filter-bank output for ONE palm image.

Panels:  (1) the image   (2) per-patch Gabor glyphs, drawn like a HOG visualisation: in every patch of the
         encoder's grid, one short line per Gabor orientation, drawn ALONG the palm line that orientation detects,
         with opacity / thickness proportional to the response energy   (3) pixel-wise dominant orientation
         (hue = orientation, brightness = response strength) with a small orientation key.

It uses the project's real GaborBank (gabor.py) with the same settings as training, and, like training
(line_masking.line_saliency(..., border=bank.pad)), pools only over the pixels that are NOT contaminated by the
bank's zero padding.

Run from the repo root (gabor.py importable):
  python visualize_gabor.py --image /path/to/palm.jpg
  python visualize_gabor.py --image palm.jpg --glyph dominant --bg black --out gabor_palm.png
Settings that match the SA-JEPA run: --gabor_orient 8 --gabor_num_scales 3 --gabor_gamma 0.5 --gabor_gray 0
(the defaults), --img_size 112 --num_patches 8.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from gabor import GaborBank, BASE_SCALE_LADDER


def line_angle_deg(o, n_orient):
    """Visual angle (degrees, counter-clockwise from the horizontal, in [0, 180)) of the PALM LINE detected by
    orientation index o. GaborBank uses theta = pi*o/n for the WAVE direction; the line is perpendicular to it."""
    return (90.0 - 180.0 * o / n_orient) % 180.0


def orientation_energy(resp, n_orient, n_scales, per_channel, scale_idx):
    """(1, K, H, W) raw responses -> (n_orient, H, W): |response| summed over the chosen scales (and RGB channels).
    Filter order in the bank: channel-major (if per_channel), then scale, then orientation."""
    _, K, H, W = resp.shape
    nch = 3 if per_channel else 1
    assert K == nch * n_scales * n_orient, (K, nch, n_scales, n_orient)
    r = resp.abs().view(nch, n_scales, n_orient, H, W)
    return r[:, scale_idx].sum(dim=(0, 1))


def patch_energy(E, grid, border):
    """(n_orient, H, W) -> (n_orient, grid, grid): mean energy per patch over the VALID pixels only."""
    n, H, W = E.shape
    valid = torch.zeros(1, 1, H, W)
    valid[..., border:H - border, border:W - border] = 1.0
    num = F.adaptive_avg_pool2d(E[None] * valid, (grid, grid))[0]
    den = F.adaptive_avg_pool2d(valid, (grid, grid))[0]
    P = num / den.clamp_min(1e-6)
    return torch.where(den > 1e-3, P, torch.zeros_like(P))


def glyph_segments(Pn, ps, mode, min_frac):
    """Pn: (n_orient, grid, grid) weights in [0, 1]. -> segments (x0, y0, x1, y1), RGBA-alpha / width arrays."""
    n, g, _ = Pn.shape
    segs, alphas, widths = [], [], []
    for r in range(g):
        for c in range(g):
            cx, cy = (c + 0.5) * ps, (r + 0.5) * ps
            os_ = range(n) if mode == "hog" else [int(Pn[:, r, c].argmax())]
            for o in os_:
                w = float(Pn[o, r, c])
                if w < min_frac:
                    continue
                th = math.pi * o / n
                dx, dy = -math.sin(th), math.cos(th)                  # LINE direction (image coords, y down)
                half = 0.45 * ps * (1.0 if mode == "hog" else 0.35 + 0.65 * w)
                segs.append([(cx - half * dx, cy - half * dy), (cx + half * dx, cy + half * dy)])
                alphas.append(min(1.0, 0.15 + 0.85 * w))
                widths.append(0.5 + 1.8 * w)
    return segs, np.array(alphas), np.array(widths)


def main():
    ap = argparse.ArgumentParser(description="HOG-style visualisation of the Gabor filter-bank output (one palm image)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="gabor_glyphs.png")
    ap.add_argument("--img_size", type=int, default=112)
    ap.add_argument("--num_patches", type=int, default=8)
    ap.add_argument("--gabor_orient", type=int, default=8)
    ap.add_argument("--gabor_num_scales", type=int, default=3)
    ap.add_argument("--gabor_gamma", type=float, default=0.5)
    ap.add_argument("--gabor_gray", type=int, default=0, help="1 = grayscale bank, 0 = per-channel (SA-JEPA run)")
    ap.add_argument("--scale_idx", type=int, nargs="*", default=None,
                    help="scales to use, 0 = finest ... (default: all scales, summed)")
    ap.add_argument("--glyph", choices=["hog", "dominant"], default="hog",
                    help="hog: every orientation of every patch (like HOG); dominant: one line per patch")
    ap.add_argument("--norm", choices=["global", "patch"], default="global",
                    help="global: divide by the strongest patch response (weak patches stay faint); "
                         "patch: divide by the patch's own maximum (orientation shape only)")
    ap.add_argument("--power", type=float, default=1.0, help="exponent on the weights (<1 boosts weak responses)")
    ap.add_argument("--min_frac", type=float, default=0.08, help="do not draw glyph lines weaker than this")
    ap.add_argument("--bg", choices=["image", "black"], default="image", help="background of the glyph panel")
    ap.add_argument("--no_map", action="store_true", help="omit the orientation-map panel")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    g, S = a.num_patches, a.img_size
    if S % g:
        raise SystemExit(f"--img_size {S} must be divisible by --num_patches {g}")
    ps = S // g
    scales = BASE_SCALE_LADDER[:a.gabor_num_scales]
    scale_idx = a.scale_idx if a.scale_idx else list(range(len(scales)))
    if any(i < 0 or i >= len(scales) for i in scale_idx):
        raise SystemExit(f"--scale_idx must be in 0..{len(scales) - 1}")

    img = Image.open(a.image).convert("RGB").resize((S, S), Image.BILINEAR)
    x01 = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    x = (x01 - 0.5) / 0.5                                            # same Normalize as dataset.py
    per_channel = not bool(a.gabor_gray)
    bank = GaborBank(n_orient=a.gabor_orient, scales=scales, gamma=a.gabor_gamma, per_channel=per_channel)
    if 2 * bank.pad >= S:
        raise SystemExit(f"image too small ({S}px) for the bank's padding band ({bank.pad}px)")
    with torch.no_grad():
        resp = bank(x)
        E = orientation_energy(resp, a.gabor_orient, len(scales), per_channel, scale_idx)     # (n, H, W)
        P = patch_energy(E, g, bank.pad)                                                       # (n, g, g)
    print(f"bank: {a.gabor_orient} orientations x {len(scales)} scales "
          f"({'per-channel' if per_channel else 'gray'}), K={bank.K}, border band = {bank.pad}px, "
          f"scales used: {scale_idx}")

    if a.norm == "global":
        Pn = P / (P.max() + 1e-8)
    else:
        Pn = P / (P.max(dim=0, keepdim=True).values + 1e-8)
    Pn = Pn.clamp(0, 1) ** a.power
    segs, alphas, widths = glyph_segments(Pn.numpy(), ps, a.glyph, a.min_frac)

    ncols = 2 if a.no_map else 3
    fig, ax = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.5))
    ax = np.atleast_1d(ax)
    arr = np.asarray(img)
    ax[0].imshow(arr, extent=(0, S, S, 0))
    ax[0].set_title("Image", fontsize=10)

    if a.bg == "black":
        ax[1].imshow(np.zeros_like(arr), extent=(0, S, S, 0))
        col = np.ones((len(segs), 4))
    else:
        ax[1].imshow((arr * 0.45).astype(np.uint8), extent=(0, S, S, 0))
        col = np.tile(np.array([1.0, 0.95, 0.3, 1.0]), (len(segs), 1))                         # yellow on dimmed image
    if segs:
        col[:, 3] = alphas
        ax[1].add_collection(LineCollection(segs, colors=col, linewidths=widths, capstyle="round"))
    ax[1].set_xlim(0, S)
    ax[1].set_ylim(S, 0)
    ax[1].set_title("Gabor orientation energy per patch", fontsize=10)

    if not a.no_map:
        n = a.gabor_orient
        with torch.no_grad():
            strength, dom = E.max(dim=0)
        s = strength.numpy()
        inner = np.zeros_like(s, dtype=bool)
        inner[bank.pad:S - bank.pad, bank.pad:S - bank.pad] = True
        s = np.where(inner, s, 0.0)
        s = np.clip(s / (np.percentile(s[inner], 99) + 1e-8), 0, 1) ** 0.7
        hsv = np.stack([dom.numpy() / n, np.ones_like(s), s], axis=-1)
        ax[2].imshow(hsv_to_rgb(hsv), extent=(0, S, S, 0))
        ax[2].set_title("Dominant line orientation", fontsize=10)
        key = ax[2].inset_axes([0.05, -0.13, 0.9, 0.06])
        key.imshow(hsv_to_rgb(np.stack([np.arange(n) / n, np.ones(n), np.ones(n)], -1))[None], aspect="auto")
        key.set_yticks([])
        step = max(1, n // 4)
        key.set_xticks(range(0, n, step))
        key.set_xticklabels([f"{line_angle_deg(o, n):.0f}°" for o in range(0, n, step)], fontsize=7)
        key.set_xlabel("palm-line angle from horizontal", fontsize=7, labelpad=1)

    for a_ in ax:
        a_.set_xticks([])
        a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.05)
    pdf = os.path.splitext(a.out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {a.out} (+ {pdf})")


if __name__ == "__main__":
    main()
