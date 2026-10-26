"""simulate_domain_shifts.py -- pixel-level simulation of palm-print domain shifts, in three groups.

  device    camera / ISP differences (iPhone vs Huawei ...): color rendering, resolution, sharpening/smoothing,
            local contrast (HDR), sensor noise, JPEG, and a 'device_profile' that chains them with CORRELATED traits
  lighting  exposure, side light, flash glare, cast shadow, colored light, haze / low contrast
  surface   palm condition: wet, oily/sweaty, dry/flaky, dirt, ballpoint-ink text   (wet + ink adapted from augment.py)

Every operator is   op(img, rng, severity, skin) -> img      img: float32 (H, W, 3) in [0, 1]
  severity in [0, 1] scales the magnitude; each call draws its own random magnitude (0.5..1) * severity.
  All spatial sizes are RELATIVE to the image width, so the same code works for 112 px ROIs and 512 px images.
  skin: (H, W) float mask in {0, 1}. Palm-ROI crops are all skin (default). For whole-hand photos on a white
  background use --skin nonwhite (the rule of augment.py: min(R,G,B) < 235).

Physical order used by `random_shift(..., group="all")`:  surface (the palm) -> lighting -> device (the camera).

Usage (writes one figure per group + an overview):
  python simulate_domain_shifts.py --image_dir /path/CASIA-MS-ROI --k 4 --size 224 --severity 0.8 --out_dir domain_shift_figs
  python simulate_domain_shifts.py --images a.png b.png --groups device surface
  python simulate_domain_shifts.py --list_ops
  # ONE figure: row 0 = originals, then one row per shift applied to the same images
  python simulate_domain_shifts.py --image_dir /path/XJTU-UP --k 6 --layout rows \
      --shifts poor_lighting strong_lighting wet device_change
Use in training code (PIL in, PIL out):
  from simulate_domain_shifts import shift_pil;  img = shift_pil(img, group="all", severity=0.6, seed=None)
Only numpy, Pillow and matplotlib are needed.
"""
import argparse
import glob
import io
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SENTENCES = ["meet me after class", "don't forget the keys", "call mom tonight", "room 214 east wing",
             "be kind to yourself", "coffee at noon?", "trust the process", "left palm sample A",
             "see you Thursday", "keep this secret", "bring the notebook", "almost home free",
             "remember to smile", "parking lot B-3", "you are enough"]


# ═══════════════════════════ helpers ═══════════════════════════
def to_float(pil):
    return np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0


def to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def lum(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def blur(a, sigma):
    """Separable Gaussian blur of an (H, W) or (H, W, C) float array; edge-replicated."""
    if sigma < 0.3:
        return a
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n, acc = out.shape[axis], np.zeros_like(out)
        for i, kv in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            acc += kv * p[tuple(sl)]
        out = acc
    return out


def resize_f(a, size, resample):
    """Resize a float (H, W, 3) array to size=(w, h) channel-wise (PIL 'F' mode keeps precision)."""
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c])).resize(size, resample))
                     for c in range(3)], axis=-1)


def mag(rng, s):
    return float(s) * float(rng.uniform(0.5, 1.0))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def skin_mask(img, mode="all"):
    if mode == "nonwhite":
        return (np.min(img, axis=2) < 235 / 255.0).astype(np.float32)
    return np.ones(img.shape[:2], dtype=np.float32)


# ═══════════════════════════ DEVICE (camera / ISP) ═══════════════════════════
def _cast(img, t, mix, sat):
    out = img.copy()
    out[..., 0] *= 1 + t
    out[..., 2] *= 1 - t
    out = out @ (np.eye(3, dtype=np.float32) + mix).T
    g = lum(out)[..., None]
    return np.clip(g + (out - g) * sat, 0, 1)


def _res(img, f):
    H, W = img.shape[:2]
    if f <= 1.05:
        return img
    small = resize_f(img, (max(8, int(round(W / f))), max(8, int(round(H / f)))), Image.BILINEAR)
    return np.clip(resize_f(small, (W, H), Image.BICUBIC), 0, 1)


def _unsharp(img, amount, sigma):
    return np.clip(img + amount * (img - blur(img, sigma)), 0, 1)


def _local_contrast(img, g, sigma):
    d = lum(img) - blur(lum(img), sigma)
    return np.clip(img + g * d[..., None], 0, 1)


def _noise(img, rng, a, b):
    """Poisson-Gaussian-like: std = sqrt(a * x + b); luma noise + weaker independent chroma noise."""
    H, W = img.shape[:2]
    std = np.sqrt(a * np.clip(lum(img), 0, 1) + b)[..., None]
    n = rng.standard_normal((H, W, 1)) * 1.0 + rng.standard_normal((H, W, 3)) * 0.6
    return np.clip(img + std * n.astype(np.float32) / 1.2, 0, 1)


def _jpeg(img, q):
    buf = io.BytesIO()
    to_pil(img).save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return to_float(Image.open(buf))


def op_color_cast(img, rng, s, skin):
    m = mag(rng, s)
    return _cast(img, float(rng.choice([-1, 1])) * 0.14 * m, rng.uniform(-1, 1, (3, 3)).astype(np.float32) * 0.08 * m,
                 1.0 + float(rng.uniform(-0.3, 0.2)) * m)


def op_resolution(img, rng, s, skin):
    return _res(img, 1.0 + 1.2 * mag(rng, s))


def op_detail(img, rng, s, skin):
    W = img.shape[1]
    m = mag(rng, s)
    if rng.random() < 0.5:
        return np.clip(blur(img, 0.012 * W * m), 0, 1)            # smooth / denoised ISP
    return _unsharp(img, 1.5 * m, 0.006 * W)                      # over-sharpened ISP


def op_local_contrast(img, rng, s, skin):
    return _local_contrast(img, 0.8 * mag(rng, s), 0.07 * img.shape[1])


def op_sensor_noise(img, rng, s, skin):
    m = mag(rng, s)
    return _noise(img, rng, 0.0015 * m, 0.0002 * m)


def op_jpeg(img, rng, s, skin):
    return _jpeg(img, round(92 - 55 * mag(rng, s)))


def op_device_profile(img, rng, s, skin):
    """One 'phone': correlated traits drive the whole ISP chain (cast -> resolution -> sharpen/smooth ->
    local contrast -> noise -> JPEG). Real devices differ in COMBINATIONS, not in independent random effects."""
    W = img.shape[1]
    warm, sharp = rng.uniform(-1, 1), rng.uniform(-1, 1)
    noisy, lowres, comp, hdr = (rng.uniform(0, 1) for _ in range(4))
    out = _cast(img, 0.12 * warm * s, rng.uniform(-1, 1, (3, 3)).astype(np.float32) * 0.06 * s,
                1.0 + 0.25 * rng.uniform(-1, 0.6) * s)
    out = _res(out, 1.0 + 1.0 * lowres * s)
    out = _unsharp(out, 1.4 * sharp * s, 0.006 * W) if sharp > 0 else np.clip(blur(out, 0.012 * W * -sharp * s), 0, 1)
    out = _local_contrast(out, 0.8 * hdr * s, 0.07 * W)
    out = _noise(out, rng, 0.0015 * noisy * s, 0.0002 * noisy * s)
    return _jpeg(out, round(92 - 55 * comp * s))


# ═══════════════════════════ LIGHTING ═══════════════════════════
def _coords(img):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)


def op_dim(img, rng, s, skin):
    m = mag(rng, s)
    out = np.clip(img, 1e-4, 1) ** (1 + 0.3 * m) * (1 - 0.65 * m)
    return _noise(out, rng, 0.003 * m, 0.0003 * m)                # underexposure amplifies sensor noise


def op_overexpose(img, rng, s, skin):
    y = img * (1 + 0.9 * mag(rng, s)) + 0.02
    return np.clip(np.where(y > 0.8, 0.8 + 0.2 * np.tanh((y - 0.8) / 0.2), y), 0, 1)     # filmic shoulder


def op_side_light(img, rng, s, skin):
    m = mag(rng, s)
    th = rng.uniform(0, 2 * math.pi)
    x, y = _coords(img)
    u = x * math.cos(th) + y * math.sin(th)
    L = np.clip(1 + 0.65 * m * u, 0.2, None)[..., None]
    out = img * L
    out[..., 0] *= 1 + 0.06 * m * u                               # light side slightly warmer
    out[..., 2] *= 1 - 0.06 * m * u
    return np.clip(out, 0, 1)


def op_glare(img, rng, s, skin):
    m = mag(rng, s)
    H, W = img.shape[:2]
    x, y = _coords(img)
    add = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        cx, cy = rng.uniform(-0.7, 0.7, 2)
        sg = rng.uniform(0.15, 0.45)
        add += rng.uniform(0.6, 1.0) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sg ** 2))
    return np.clip(img + 0.65 * m * add[..., None] * np.array([1, 1, 0.95], np.float32), 0, 1)


def op_cast_shadow(img, rng, s, skin):
    m = mag(rng, s)
    th = rng.uniform(0, 2 * math.pi)
    x, y = _coords(img)
    u = x * math.cos(th) + y * math.sin(th)
    c = rng.uniform(-0.3, 0.5)
    pen = 0.06
    if rng.random() < 0.5:                                        # finger / hand shadow: a soft stripe
        w = rng.uniform(0.15, 0.35)
        sh = sigmoid((u - (c - w)) / pen) * sigmoid(((c + w) - u) / pen)
    else:                                                         # half-plane shadow
        sh = sigmoid((u - c) / pen)
    out = img * (1 - 0.6 * m * sh)[..., None]
    out[..., 2] *= 1 + 0.08 * m * sh                              # shadows are lit by bluish sky light
    return np.clip(out, 0, 1)


def op_colored_light(img, rng, s, skin):
    vec = np.array(list(({"warm": (1.18, 1.0, 0.78), "cool": (0.85, 1.0, 1.18),
                          "green": (0.92, 1.07, 0.92), "magenta": (1.08, 0.92, 1.06)})[
        str(rng.choice(["warm", "cool", "green", "magenta"]))]), np.float32)
    return np.clip(img * (1 + (vec - 1) * mag(rng, s) * 1.3), 0, 1)


def op_haze(img, rng, s, skin):
    m = mag(rng, s)
    return np.clip((img - 0.5) * (1 - 0.5 * m) + 0.5 + 0.10 * m, 0, 1)    # lifted blacks, low contrast


# ═══════════════════════════ SURFACE (palm condition) ═══════════════════════════
def _split(img, radius):
    low = blur(img, radius)
    return low, img - low


def op_wet(img, rng, s, skin):
    """Damp skin: crease pooling + soft sheen (adapted from augment.edit_wet_pixel; radii scale with width)."""
    k = img.shape[1] / 512.0
    a = 0.5 + mag(rng, s)
    low, detail = _split(img, 9 * k)
    shade = lum(low)
    sel = skin > 0.5
    lo_p, hi_p = np.percentile(shade[sel], [25, 70]) if sel.any() else (0.3, 0.6)
    crease = blur(np.clip((hi_p - shade) / (hi_p - lo_p + 1e-6), 0, 1) * skin, 1.8 * k) ** 1.1
    raised = blur(np.clip((shade - lo_p) / (hi_p - lo_p + 1e-6), 0, 1) * skin, 8 * k)
    sheen = blur((raised ** 1.6) * float(rng.uniform(0.85, 1.0)), 3 * k)
    out = low * (1 - 0.14 * a * crease[..., None])
    out[..., 2] += 0.03 * a * crease
    out = np.clip(out + 0.07 * a * sheen[..., None], 0, 1)
    out[..., 2] = np.clip(out[..., 2] + 0.015 * a * sheen, 0, 1)
    edge = np.abs(shade - blur(shade, 2.5 * k)) * skin
    edge = edge / (edge.max() + 1e-6)
    streaks = blur((edge > 0.35).astype(np.float32) * skin, 0.8 * k) * sheen
    out = np.clip(out + 0.05 * a * streaks[..., None], 0, 1)
    return np.clip(out + detail, 0, 1)


def op_oily(img, rng, s, skin):
    """Sweaty / oily: broad specular sheen that also washes out the fine line contrast."""
    k = img.shape[1] / 512.0
    m = mag(rng, s)
    low, detail = _split(img, 6 * k)
    shade = lum(low)
    sel = skin > 0.5
    lo_p, hi_p = np.percentile(shade[sel], [25, 80]) if sel.any() else (0.3, 0.6)
    raised = blur(np.clip((shade - lo_p) / (hi_p - lo_p + 1e-6), 0, 1) * skin, 5 * k)
    sheen = blur(raised ** 1.3, 4 * k) * float(rng.uniform(0.8, 1.0))
    out = low + detail * (1 - 0.35 * m)
    return np.clip(out + 0.22 * m * sheen[..., None] * np.array([1, 1, 0.95], np.float32), 0, 1)


def op_dry(img, rng, s, skin):
    """Dry / flaky skin: whitish flakes, ashy desaturation, rougher fine texture."""
    H, W = img.shape[:2]
    m = mag(rng, s)
    f = blur(rng.standard_normal((H, W)).astype(np.float32), max(0.8, 0.005 * W))
    f = f / (f.std() + 1e-6)
    flake = sigmoid((f - 1.3) * 3.0) * skin
    low, detail = _split(img, max(1.0, 0.012 * W))
    out = low + detail * (1 + 0.4 * m)
    g = lum(out)[..., None]
    out = g + (out - g) * (1 - 0.2 * m)
    return np.clip(out + (1 - out) * (0.38 * m) * flake[..., None], 0, 1)


def op_dirt(img, rng, s, skin):
    """Soft brownish smudges / dirt on the palm."""
    H, W = img.shape[:2]
    m = mag(rng, s)
    nz = blur(rng.standard_normal((H, W)).astype(np.float32), 0.05 * W)
    nz = nz / (nz.std() + 1e-6)
    blot = (sigmoid((nz - 0.6) * 4.0) * skin)[..., None] * 0.55 * m
    return np.clip(img * (1 - blot) + img * np.array([0.62, 0.50, 0.40], np.float32) * blot, 0, 1)


def _find_font(path):
    if path and os.path.isfile(path):
        return path
    try:
        from matplotlib import font_manager
        return font_manager.findfont("DejaVu Sans")
    except Exception:
        return None


def op_ink_text(img, rng, s, skin, font_path=None):
    """Ballpoint-ink sentence blended into the palm (adapted from augment.overlay_handwritten_note)."""
    H, W = img.shape[:2]
    k = W / 512.0
    text = str(rng.choice(SENTENCES))
    size = max(10, int(W * rng.uniform(0.09, 0.15)))
    fp = _find_font(font_path)
    try:
        font = ImageFont.truetype(fp, size)
    except Exception:
        font = ImageFont.load_default()
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words, lines, cur = text.split(), [], text.split()[0]
    for w_ in words[1:]:
        if dummy.textbbox((0, 0), f"{cur} {w_}", font=font)[2] <= 0.85 * W:
            cur = f"{cur} {w_}"
        else:
            lines.append(cur)
            cur = w_
    wrapped = "\n".join(lines + [cur])
    bbox = dummy.multiline_textbbox((0, 0), wrapped, font=font, spacing=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(4, int(0.055 * W))
    ink = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(ink)
    pos = (pad - bbox[0], pad - bbox[1])
    d.multiline_text(pos, wrapped, font=font, fill=(12, 40, 130, 220), spacing=2)
    d.multiline_text((pos[0] + 1, pos[1]), wrapped, font=font, fill=(10, 32, 110, 120), spacing=2)
    ink = ink.filter(ImageFilter.GaussianBlur(radius=max(0.3, 0.45 * k * 2)))
    ink = ink.rotate(float(rng.uniform(-18, 12)), expand=True, resample=Image.Resampling.BICUBIC)
    arr = np.asarray(ink).astype(np.float32)
    ih, iw = arr.shape[:2]
    yy, xx = np.mgrid[0:ih, 0:iw]
    amp = max(1.5, min(ih, iw) * 0.035)
    wl1, wl2 = max(4.0, 14 * k * 2), max(4.0, 16 * k * 2)
    xw = np.clip(xx + amp * np.sin(yy / wl1 + rng.uniform(0, 6)), 0, iw - 1).astype(np.int32)
    yw = np.clip(yy + amp * 0.8 * np.sin(xx / wl2 + rng.uniform(0, 6)), 0, ih - 1).astype(np.int32)
    ink = Image.fromarray(np.clip(arr[yw, xw], 0, 255).astype(np.uint8), mode="RGBA")
    if skin.mean() > 0.95:
        cx, cy = W // 2, H // 2
    else:
        ys, xs = np.where(skin > 0.5)
        cy, cx = (int(np.percentile(ys, 42)), int(np.percentile(xs, 50))) if len(ys) else (H // 2, W // 2)
    x = int(np.clip(cx - ink.size[0] // 2 + rng.integers(-W // 18, W // 18 + 1), 0, max(0, W - ink.size[0])))
    y = int(np.clip(cy - ink.size[1] // 2 + rng.integers(-H // 18, H // 18 + 1), 0, max(0, H - ink.size[1])))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.paste(ink, (x, y), ink)
    alpha = np.asarray(layer.split()[-1], dtype=np.float32) / 255.0 * blur(skin, 1.2)
    crease = np.clip((0.65 - lum(img)) / 0.45, 0, 1)
    alpha = np.clip(alpha * (0.75 + 0.45 * crease), 0, 1)[..., None] * (0.55 + 0.45 * min(1.0, s / 0.8))
    tint = np.array([0.05, 0.18, 0.62], np.float32)
    return np.clip(img * (1 - 0.95 * alpha) + img * tint * (0.95 * alpha), 0, 1)


OPS = {
    "device": {"color_cast": op_color_cast, "resolution": op_resolution, "detail": op_detail,
               "local_contrast": op_local_contrast, "sensor_noise": op_sensor_noise, "jpeg": op_jpeg,
               "device_profile": op_device_profile},
    "lighting": {"dim": op_dim, "overexpose": op_overexpose, "side_light": op_side_light, "glare": op_glare,
                 "cast_shadow": op_cast_shadow, "colored_light": op_colored_light, "haze": op_haze},
    "surface": {"wet": op_wet, "oily": op_oily, "dry": op_dry, "dirt": op_dirt, "ink_text": op_ink_text},
}


# ═══════════════════════════ composition (also usable as a training augmentation) ═══════════════════════════
def random_shift(img, rng, group="all", severity=0.6, skin=None, p_extra=0.3):
    """One random domain shift. group: device | lighting | surface | all.
       device -> one 'device_profile';  lighting -> 1 operator (a 2nd with prob p_extra);  surface -> same;
       all -> surface, then lighting, then device (the physical order)."""
    skin = skin_mask(img) if skin is None else skin
    if group == "all":
        for g in ("surface", "lighting", "device"):
            img = random_shift(img, rng, g, severity, skin, p_extra)
        return img
    if group == "device":
        return OPS["device"]["device_profile"](img, rng, severity, skin)
    names = list(OPS[group])
    n = 2 if rng.random() < p_extra else 1
    for name in rng.choice(names, size=n, replace=False):
        img = OPS[group][str(name)](img, rng, severity, skin)
    return img


def shift_pil(pil_img, group="all", severity=0.6, seed=None, skin="all"):
    """PIL in -> PIL out. seed=None gives a fresh random shift every call."""
    img = to_float(pil_img)
    return to_pil(random_shift(img, np.random.default_rng(seed), group, severity, skin_mask(img, skin)))


# ═══════════════════════════ images / figure ═══════════════════════════
def list_images(root):
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(IMG_EXTS))


def load_image(path, size, fit):
    im = Image.open(path).convert("RGB")
    if fit == "contain":                                          # whole hand: keep aspect on a white canvas
        w, h = im.size
        sc = size / max(w, h)
        im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
        return canvas
    return im.resize((size, size), Image.LANCZOS)


def pick_paths(a):
    if a.images:
        return list(a.images)
    found = list_images(a.image_dir)
    if len(found) < a.k:
        raise SystemExit(f"only {len(found)} images under {a.image_dir}, need {a.k}")
    if a.pick == "random":
        idx = sorted(np.random.RandomState(a.seed).choice(len(found), a.k, replace=False).tolist())
    else:
        idx = np.linspace(0, len(found) - 1, a.k).round().astype(int)
    return [found[i] for i in idx]


def draw_grid(rows, col_titles, path, dpi, cell=1.6):
    """rows: list (per sample) of lists of float images; first column = original."""
    K, C = len(rows), len(rows[0])
    fig, ax = plt.subplots(K, C, figsize=(cell * C, cell * K + 0.45), squeeze=False)
    for i in range(K):
        for j in range(C):
            ax[i, j].imshow(np.clip(rows[i][j], 0, 1))
            ax[i, j].axis("off")
    for j, t in enumerate(col_titles):
        ax[0, j].set_title(t.replace("_", " "), fontsize=8)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93 if K > 1 else 0.88, bottom=0.005, wspace=0.03, hspace=0.04)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {path} (+ .pdf)")


SHIFT_ALIASES = {"poor_lighting": ("lighting", "dim"), "strong_lighting": ("lighting", "overexpose"),
                 "device_change": ("device", "device_profile"), "wet_palm": ("surface", "wet")}
DEFAULT_SHIFTS = ["poor_lighting", "strong_lighting", "side_light", "glare", "cast_shadow",
                  "wet", "dry", "device_change"]


def resolve_shift(name):
    """'poor_lighting' | 'dim' | ... -> (group, op name)."""
    if name in SHIFT_ALIASES:
        return SHIFT_ALIASES[name]
    for g, d in OPS.items():
        if name in d:
            return g, name
    raise SystemExit(f"unknown shift '{name}'. Aliases: {list(SHIFT_ALIASES)}; operators: "
                     f"{[n for d in OPS.values() for n in d]}")


def draw_rows(origs, skins, shifts, severity, seed, font, path, dpi, cell=1.5, labels=True):
    """Row 0 = the original images; every other row = ONE shift applied to the SAME images (columns)."""
    K, R = len(origs), len(shifts) + 1
    fig, ax = plt.subplots(R, K, figsize=(cell * K, cell * R), squeeze=False)
    names = ["original"] + [n.replace("_", " ") for n in shifts]
    for j in range(K):
        ax[0, j].imshow(origs[j])
    for r, name in enumerate(shifts, start=1):
        g, op = resolve_shift(name)
        for j in range(K):
            rng = np.random.default_rng([seed, j, r])
            f = OPS[g][op]
            out = f(origs[j], rng, severity, skins[j], font_path=font) if op == "ink_text" \
                else f(origs[j], rng, severity, skins[j])
            ax[r, j].imshow(np.clip(out, 0, 1))
    for r in range(R):
        for j in range(K):
            ax[r, j].axis("off")
        if labels:
            ax[r, 0].text(-0.05, 0.5, names[r], transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.005, wspace=0.03, hspace=0.04)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {path} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description="Simulate device / lighting / surface domain shifts on palm images")
    ap.add_argument("--image_dir", default="/home/pai-ng/Jamal/CASIA-MS-ROI")
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=4, help="number of original palm images (rows)")
    ap.add_argument("--pick", choices=["spread", "random"], default="spread")
    ap.add_argument("--size", type=int, default=224, help="working / display size in pixels")
    ap.add_argument("--fit", choices=["resize", "contain"], default="resize",
                    help="resize = ROI crops (default); contain = whole-hand photos on a white canvas")
    ap.add_argument("--skin", choices=["all", "nonwhite"], default="all")
    ap.add_argument("--severity", type=float, default=0.8)
    ap.add_argument("--groups", nargs="+", default=["device", "lighting", "surface", "overview"],
                    choices=["device", "lighting", "surface", "overview"])
    ap.add_argument("--ops", nargs="*", default=None, help="restrict the per-group figures to these operator names")
    ap.add_argument("--font", default=None, help="TTF for ink_text (default: matplotlib's DejaVu Sans)")
    ap.add_argument("--out_dir", default="domain_shift_figs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--layout", choices=["columns", "rows"], default="columns",
                    help="columns: one figure per group (operators as columns). rows: ONE figure, first row = originals, "
                         "every other row = one named shift applied to the same images (use with --shifts)")
    ap.add_argument("--shifts", nargs="*", default=None,
                    help="rows layout: shift names, e.g. poor_lighting strong_lighting wet device_change glare ... "
                         f"(aliases {list(SHIFT_ALIASES)} or any operator name). Default: {' '.join(DEFAULT_SHIFTS)}")
    ap.add_argument("--no_labels", action="store_true", help="rows layout: omit the row labels")
    ap.add_argument("--list_ops", action="store_true")
    a = ap.parse_args()
    if a.list_ops:
        for g, d in OPS.items():
            print(f"{g:9s}: {', '.join(d)}")
        return
    os.makedirs(a.out_dir, exist_ok=True)
    paths = pick_paths(a)
    print("images:", *paths, sep="\n  ")
    origs = [to_float(load_image(p, a.size, a.fit)) for p in paths]
    skins = [skin_mask(o, a.skin) for o in origs]

    if a.layout == "rows" or a.shifts:
        draw_rows(origs, skins, a.shifts or DEFAULT_SHIFTS, a.severity, a.seed, a.font,
                  os.path.join(a.out_dir, "domain_shift_rows.png"), a.dpi, labels=not a.no_labels)
        return

    for g in ("device", "lighting", "surface"):
        if g not in a.groups:
            continue
        names = [n for n in OPS[g] if not a.ops or n in a.ops]
        if not names:
            continue
        rows = []
        for i, (o, sk) in enumerate(zip(origs, skins)):
            row = [o]
            for j, n in enumerate(names):
                rng = np.random.default_rng([a.seed, i, list(OPS[g]).index(n)])
                op = OPS[g][n]
                row.append(op(o, rng, a.severity, sk, font_path=a.font) if n == "ink_text"
                           else op(o, rng, a.severity, sk))
            rows.append(row)
        draw_grid(rows, ["original"] + names, os.path.join(a.out_dir, f"domain_shift_{g}.png"), a.dpi)

    if "overview" in a.groups:
        cols = ["device", "lighting", "surface", "all"]
        rows = []
        for i, (o, sk) in enumerate(zip(origs, skins)):
            row = [o]
            for j, g in enumerate(cols):
                row.append(random_shift(o, np.random.default_rng([a.seed, i, 100 + j]), g, a.severity, sk))
            rows.append(row)
        draw_grid(rows, ["original", "device", "lighting", "surface", "all three"],
                  os.path.join(a.out_dir, "domain_shift_overview.png"), a.dpi)


if __name__ == "__main__":
    main()
