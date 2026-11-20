"""corruption_types.py -- lighting / surface / device corruption groups for --corruption_type.

Batched torch ports (GPU or CPU) of the operators in simulate_domain_shifts.py. Every operator
takes x (n,3,H,W) in [0,1] and returns the same shape; every sample gets its OWN random
strength / parameters. corruption.corrupt_images decides WHICH samples and WHICH operators.

Groups (registered in GROUPS, applied to the selected samples exactly like the 7 'current' ops):
  lighting : light_knob (signed poor<->strong brightness, mild range), side_light, glare,
             cast_shadow, colored_light, haze
  surface  : wet, dry, dirt
  device   : dev_color_cast, dev_resolution, dev_detail, dev_local_contrast, dev_sensor_noise,
             dev_jpeg, dev_device_profile
Config (all optional, getattr defaults):  corruption_severity (0.6), lighting_knob_max (0.4).
"""
import io
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

LIGHTING = ["light_knob", "side_light", "glare", "cast_shadow", "colored_light", "haze"]
SURFACE = ["wet", "dry", "dirt"]
LEGACY = ["wet_sheen", "wet_film"]            # alternative wet operators: not in any group, reachable by name
DEVICE = ["dev_color_cast", "dev_resolution", "dev_detail", "dev_local_contrast",
          "dev_sensor_noise", "dev_jpeg", "dev_device_profile"]


# ───────────────────────────── helpers ─────────────────────────────
def _u(n, x, lo=0.0, hi=1.0):
    return lo + (hi - lo) * torch.rand(n, 1, 1, 1, device=x.device, dtype=x.dtype)


def _mag(n, x, s):
    """per-sample strength = severity * U(0.5, 1)  (as in simulate_domain_shifts.mag)"""
    return float(s) * _u(n, x, 0.5, 1.0)


def _lum(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _gblur(x, sigma):
    sigma = float(sigma)
    if sigma < 0.3:
        return x
    k = 2 * int(math.ceil(3 * sigma)) + 1
    k = min(k, 2 * ((min(x.shape[-2:]) - 1) // 2) - 1)          # reflect padding needs k//2 < size
    return TF.gaussian_blur(x, [k, k], [sigma, sigma])


def _sig(x):
    return torch.sigmoid(x)


def _coords(x):
    H, W = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                            torch.arange(W, device=x.device, dtype=x.dtype), indexing="ij")
    return ((xx - W / 2) / (W / 2)).view(1, 1, H, W), ((yy - H / 2) / (H / 2)).view(1, 1, H, W)


def _noise(x, a, b):
    """Poisson-Gaussian-like noise: std = sqrt(a*lum + b); luma + weaker chroma noise."""
    n, _, H, W = x.shape
    std = torch.sqrt(a * _lum(x).clamp(0, 1) + b)
    nz = torch.randn(n, 1, H, W, device=x.device, dtype=x.dtype) + 0.6 * torch.randn_like(x)
    return (x + std * nz / 1.2).clamp(0, 1)


def _cast(x, t, mix, sat):
    out = torch.cat([x[:, 0:1] * (1 + t), x[:, 1:2], x[:, 2:3] * (1 - t)], dim=1)
    eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
    out = torch.einsum("bij,bjhw->bihw", eye + mix, out)
    g = _lum(out)
    return (g + (out - g) * sat).clamp(0, 1)


def _res_one(xi, f):
    """xi (1,3,H,W): down by factor f (bilinear) then back up (bicubic)."""
    H, W = xi.shape[-2:]
    if f <= 1.05:
        return xi
    small = F.interpolate(xi, size=(max(8, int(round(H / f))), max(8, int(round(W / f)))),
                          mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)


def _unsharp(x, amount, sigma):
    return (x + amount * (x - _gblur(x, sigma))).clamp(0, 1)


def _local_contrast(x, g, sigma):
    L = _lum(x)
    return (x + g * (L - _gblur(L, sigma))).clamp(0, 1)


def _jpeg_one(xi, q):
    import numpy as np
    from PIL import Image
    arr = (xi[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).to(xi.device, xi.dtype)


# ───────────────────────────── LIGHTING ─────────────────────────────
def light_knob(x, s, kmax=0.4, k=None):
    """One signed knob k ~ U(-kmax, kmax): k<0 poor lighting (dim), k>0 strong lighting (over-exposure).
    kmax is kept small on purpose so images never become too dark or too bright."""
    n = x.size(0)
    k = _u(n, x, -kmax, kmax) if k is None else torch.full((n, 1, 1, 1), float(k), device=x.device, dtype=x.dtype)
    m = k.abs()
    dim = x.clamp(1e-4, 1) ** (1 + 0.3 * m) * (1 - 0.65 * m)
    dim = _noise(dim, 0.003 * m, 0.0003 * m)                    # under-exposure amplifies sensor noise
    y = x * (1 + 0.5 * m) + 0.01
    over = torch.where(y > 0.8, 0.8 + 0.2 * torch.tanh((y - 0.8) / 0.2), y).clamp(0, 1)
    return torch.where(k < 0, dim, over)


def side_light(x, s):
    n = x.size(0)
    m, th = _mag(n, x, s), _u(n, x, 0, 2 * math.pi)
    cx, cy = _coords(x)
    u = cx * torch.cos(th) + cy * torch.sin(th)
    out = x * (1 + 0.65 * m * u).clamp(min=0.2)
    out = torch.cat([out[:, 0:1] * (1 + 0.06 * m * u), out[:, 1:2], out[:, 2:3] * (1 - 0.06 * m * u)], 1)
    return out.clamp(0, 1)


def glare(x, s):
    n, _, H, W = x.shape
    m = _mag(n, x, s)
    cx, cy = _coords(x)
    add = torch.zeros(n, 1, H, W, device=x.device, dtype=x.dtype)
    for b in range(2):                                           # 1 blob always, a 2nd with prob 0.5
        on = torch.ones(n, 1, 1, 1, device=x.device, dtype=x.dtype) if b == 0 else \
            (torch.rand(n, 1, 1, 1, device=x.device) < 0.5).to(x.dtype)
        px, py, sg, amp = _u(n, x, -0.7, 0.7), _u(n, x, -0.7, 0.7), _u(n, x, 0.15, 0.45), _u(n, x, 0.6, 1.0)
        add = add + on * amp * torch.exp(-((cx - px) ** 2 + (cy - py) ** 2) / (2 * sg ** 2))
    tint = x.new_tensor([1.0, 1.0, 0.95]).view(1, 3, 1, 1)
    return (x + 0.65 * m * add * tint).clamp(0, 1)


def cast_shadow(x, s):
    n = x.size(0)
    m, th = _mag(n, x, s), _u(n, x, 0, 2 * math.pi)
    cx, cy = _coords(x)
    u = cx * torch.cos(th) + cy * torch.sin(th)
    c, pen, w = _u(n, x, -0.3, 0.5), 0.06, _u(n, x, 0.15, 0.35)
    stripe = _sig((u - (c - w)) / pen) * _sig(((c + w) - u) / pen)    # finger / hand shadow
    half = _sig((u - c) / pen)                                        # half-plane shadow
    sh = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.5, stripe, half)
    out = x * (1 - 0.6 * m * sh)
    out = torch.cat([out[:, 0:2], out[:, 2:3] * (1 + 0.08 * m * sh)], 1)   # shadows are bluish
    return out.clamp(0, 1)


_LIGHT_COLORS = [(1.18, 1.0, 0.78), (0.85, 1.0, 1.18), (0.92, 1.07, 0.92), (1.08, 0.92, 1.06)]


def colored_light(x, s):
    n = x.size(0)
    tab = x.new_tensor(_LIGHT_COLORS)
    vec = tab[torch.randint(0, len(_LIGHT_COLORS), (n,), device=x.device)].view(n, 3, 1, 1)
    return (x * (1 + (vec - 1) * _mag(n, x, s) * 1.3)).clamp(0, 1)


def haze(x, s):
    m = _mag(x.size(0), x, s)
    return ((x - 0.5) * (1 - 0.5 * m) + 0.5 + 0.10 * m).clamp(0, 1)


# ───────────────────────────── SURFACE ─────────────────────────────
def wet_film(x, s):
    """Alternative wet (not used by default): wet palm, and ONLY wet: a water film on the skin. (1) the skin gets uniformly darker and slightly richer in
    colour (water absorbs light), (2) fine micro-texture is smoothed a little (water fills it), (3) thin specular
    glints appear along the palm lines / edges, inside a patchy random water film. Nothing here depends on the
    large-scale brightness pattern of the image, so uneven lighting is NOT turned into dirt-like blotches
    (the default `wet` is the glossy operator; the crease/sheen one is wet_sheen)."""
    n, _, H, W = x.shape
    k = W / 512.0
    a = 0.5 + _mag(n, x, s)                                              # wetness, per sample
    out = x * (1 - 0.10 * a)
    g = _lum(out)
    out = g + (out - g) * (1 + 0.12 * a)                                 # darker, more saturated
    out = out + (_gblur(out, max(0.8, 1.5 * k)) - out) * (0.30 * a)      # micro-texture partly filled by water
    L = _gblur(_lum(x), max(1.5, 3.5 * k))                               # smooth first: lines, not pixel noise
    edge = (L - _gblur(L, max(3.0, 8 * k))).abs()
    edge = edge / (edge.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    glint = _gblur((edge > 0.30).to(x.dtype), max(1.0, 2.0 * k))         # thin highlights along lines / edges
    film = _sig((_unit_noise(x, 0.05 * W) - 0.2) * 3.0)                  # patchy water film
    tint = x.new_tensor([1.0, 1.0, 0.97]).view(1, 3, 1, 1)
    return (out + 0.22 * a * glint * film * tint).clamp(0, 1)


def _rand_field(n, H, W, x):
    return torch.rand(n, 1, H, W, device=x.device, dtype=x.dtype)


def wet(x, s):
    """Wet palm = simulate_domain_shifts.op_wet_gloss (wet_look.py), torch port with the same constants:
    wet tone (darker, richer colour, a little more contrast) + a fake normal map from the image's own brightness +
    Blinn-Phong specular glints that follow the creases, modulated by an uneven random water film + a soft sheen.
    Sizes scale with W/627 as in the original. Alternatives: wet_sheen (older default), wet_film."""
    n, _, H, W = x.shape
    sc = W / 627.0
    k = 0.5 + _mag(n, x, s)                                              # strength factor
    mx, mn = x.amax(1, keepdim=True), x.amin(1, keepdim=True)
    sat = torch.where(mx > 1e-6, (mx - mn) / mx.clamp(min=1e-6), torch.zeros_like(mx))
    new_mx = mx * (1 - 0.12 * k)                                         # darker
    new_mn = new_mx * (1 - (sat * (1 + 0.35 * k)).clamp(0, 1))           # richer colour (HSV saturation x(1+0.35k))
    t = torch.where(mx > mn, (mx - x) / (mx - mn).clamp(min=1e-6), torch.zeros_like(x))
    base = new_mx - t * (new_mx - new_mn)
    base = ((base - 0.5) * (1 + 0.12 * k) + 0.5).clamp(0, 1)
    height = _gblur(_lum(x), 2.0 * sc)                                   # brightness as a height map
    p = F.pad(height, (1, 1, 1, 1), mode="replicate")
    gx = ((p[..., :-2, 2:] - p[..., :-2, :-2]) + 2 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
          + (p[..., 2:, 2:] - p[..., 2:, :-2])) * sc
    gy = ((p[..., 2:, :-2] - p[..., :-2, :-2]) + 2 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
          + (p[..., 2:, 2:] - p[..., :-2, 2:])) * sc
    nrm = torch.cat([-gx * 6.0, -gy * 6.0, torch.ones_like(height)], 1)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    Ld = torch.cat([-0.45 + _u(n, x, -0.25, 0.25), -0.55 + _u(n, x, -0.25, 0.25),
                    torch.full((n, 1, 1, 1), 0.70, device=x.device, dtype=x.dtype)], 1)
    Ld = Ld / Ld.norm(dim=1, keepdim=True)
    Hv = Ld + x.new_tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
    Hv = Hv / Hv.norm(dim=1, keepdim=True)
    ndh = (nrm * Hv).sum(1, keepdim=True).clamp(0, 1)
    film = _gblur(_rand_field(n, H, W, x), 40.0 * sc)                    # uneven water layer
    f_min, f_max = film.amin(dim=(1, 2, 3), keepdim=True), film.amax(dim=(1, 2, 3), keepdim=True)
    film = (film - f_min) / (f_max - f_min + 1e-8)
    spec = ndh ** 60 * (0.5 + film)
    sheen = _gblur(ndh ** 8, 12.0 * sc) * film * 0.12 * k
    return (base + spec * 0.42 * k + sheen).clamp(0, 1)


def wet_sheen(x, s):
    """Soft wet look = simulate_domain_shifts.op_wet (from augment.py): crease pooling + soft sheen + edge streaks.
    The masks come from the image's own brightness (blotchy on unevenly lit palms). Was the default `wet` before."""
    n, _, H, W = x.shape
    k = W / 512.0
    a = 0.5 + _mag(n, x, s)
    low = _gblur(x, 9 * k)
    detail = x - low
    shade = _lum(low)
    q = torch.quantile(shade.flatten(1), shade.new_tensor([0.25, 0.70]), dim=1)      # (2, n)
    lo, hi = q[0].view(n, 1, 1, 1), q[1].view(n, 1, 1, 1)
    crease = _gblur(((hi - shade) / (hi - lo + 1e-6)).clamp(0, 1), 1.8 * k) ** 1.1
    raised = _gblur(((shade - lo) / (hi - lo + 1e-6)).clamp(0, 1), 8 * k)
    sheen = _gblur((raised ** 1.6) * _u(n, x, 0.85, 1.0), 3 * k)
    out = low * (1 - 0.14 * a * crease)
    out = torch.cat([out[:, 0:2], out[:, 2:3] + 0.03 * a * crease], 1)
    out = (out + 0.07 * a * sheen).clamp(0, 1)
    out = torch.cat([out[:, 0:2], (out[:, 2:3] + 0.015 * a * sheen).clamp(0, 1)], 1)
    edge = (shade - _gblur(shade, 2.5 * k)).abs()
    edge = edge / (edge.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    streaks = _gblur((edge > 0.35).to(x.dtype), 0.8 * k) * sheen
    return (out + 0.05 * a * streaks).add(detail).clamp(0, 1)


def _unit_noise(x, sigma):
    n, _, H, W = x.shape
    nz = _gblur(torch.randn(n, 1, H, W, device=x.device, dtype=x.dtype), sigma)
    return nz / (nz.flatten(1).std(dim=1).view(n, 1, 1, 1) + 1e-6)


def dry(x, s):
    """Dry / flaky skin: whitish flakes, ashy desaturation, rougher fine texture."""
    n, _, H, W = x.shape
    m = _mag(n, x, s)
    flake = _sig((_unit_noise(x, max(0.8, 0.005 * W)) - 1.3) * 3.0)
    low = _gblur(x, max(1.0, 0.012 * W))
    out = low + (x - low) * (1 + 0.4 * m)
    g = _lum(out)
    out = g + (out - g) * (1 - 0.2 * m)
    return (out + (1 - out) * (0.38 * m) * flake).clamp(0, 1)


def dirt(x, s):
    """Soft brownish smudges at random locations / intensities."""
    n, _, H, W = x.shape
    m = _mag(n, x, s)
    blot = _sig((_unit_noise(x, 0.05 * W) - 0.6) * 4.0) * 0.55 * m
    brown = x.new_tensor([0.62, 0.50, 0.40]).view(1, 3, 1, 1)
    return (x * (1 - blot) + x * brown * blot).clamp(0, 1)


# ───────────────────────────── DEVICE ─────────────────────────────
def dev_color_cast(x, s):
    n = x.size(0)
    m = _mag(n, x, s)
    sign = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.5, -1.0, 1.0).to(x.dtype)
    mix = (torch.rand(n, 3, 3, device=x.device, dtype=x.dtype) * 2 - 1) * 0.08 * m.view(n, 1, 1)
    return _cast(x, sign * 0.14 * m, mix, 1.0 + _u(n, x, -0.3, 0.2) * m)


def dev_resolution(x, s):
    n = x.size(0)
    f = (1.0 + 1.2 * _mag(n, x, s)).flatten().tolist()
    return torch.cat([_res_one(x[i:i + 1], f[i]) for i in range(n)], 0)


def dev_detail(x, s):
    n, _, H, W = x.shape
    m = _mag(n, x, s).flatten().tolist()
    smooth = (torch.rand(n) < 0.5).tolist()
    outs = []
    for i in range(n):
        xi = x[i:i + 1]
        outs.append(_gblur(xi, 0.012 * W * m[i]).clamp(0, 1) if smooth[i]       # smooth / denoised ISP
                    else _unsharp(xi, 1.5 * m[i], 0.006 * W))                   # over-sharpened ISP
    return torch.cat(outs, 0)


def dev_local_contrast(x, s):
    return _local_contrast(x, 0.8 * _mag(x.size(0), x, s), 0.07 * x.shape[-1])


def dev_sensor_noise(x, s):
    m = _mag(x.size(0), x, s)
    return _noise(x, 0.0015 * m, 0.0002 * m)


def dev_jpeg(x, s):
    q = (92 - 55 * _mag(x.size(0), x, s)).round().flatten().tolist()
    return torch.cat([_jpeg_one(x[i:i + 1], q[i]) for i in range(x.size(0))], 0)


def dev_device_profile(x, s):
    """One 'phone': correlated traits drive the whole ISP chain (cast -> resolution -> sharpen/smooth ->
    local contrast -> noise -> JPEG). Per sample."""
    n, _, H, W = x.shape
    outs = []
    for i in range(n):
        xi = x[i:i + 1]
        r = torch.rand(7).tolist()
        warm, sharp = r[0] * 2 - 1, r[1] * 2 - 1
        noisy, lowres, comp, hdr = r[2], r[3], r[4], r[5]
        mix = (torch.rand(1, 3, 3, device=x.device, dtype=x.dtype) * 2 - 1) * 0.06 * s
        sat = 1.0 + 0.25 * (-1 + 1.6 * r[6]) * s
        o = _cast(xi, 0.12 * warm * s, mix, sat)
        o = _res_one(o, 1.0 + 1.0 * lowres * s)
        o = _unsharp(o, 1.4 * sharp * s, 0.006 * W) if sharp > 0 else _gblur(o, 0.012 * W * -sharp * s).clamp(0, 1)
        o = _local_contrast(o, 0.8 * hdr * s, 0.07 * W)
        o = _noise(o, 0.0015 * noisy * s, 0.0002 * noisy * s)
        outs.append(_jpeg_one(o, round(92 - 55 * comp * s)))
    return torch.cat(outs, 0)


# ───────────────────────────── registry ─────────────────────────────
def _sub(fn):
    """(xs, mask, cfg) -> apply fn only to the masked samples (the other ones are untouched)."""
    def run(xs, mask, cfg):
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return xs
        out = xs.clone()
        s = getattr(cfg, "corruption_severity", 0.6)
        if fn is light_knob:
            out[idx] = light_knob(xs[idx], s, getattr(cfg, "lighting_knob_max", 0.4))
        else:
            out[idx] = fn(xs[idx], s).to(xs.dtype)
        return out
    return run


FN = {name: _sub(globals()[name]) for name in LIGHTING + SURFACE + LEGACY + DEVICE}
GROUPS = {"lighting": LIGHTING, "surface": SURFACE, "device": DEVICE}
