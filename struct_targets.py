"""struct_targets.py — alternative per-patch structure targets for the structural-anchoring ablation.

Every target is a fixed (non-learned) module with the same interface as gabor.GaborBank:

    resp = bank(images)          # (B, K, H, W), same H, W as the input
    desc = patch_energy_descriptor(resp, num_patches)   # (B, P, K): |resp| averaged per patch, L2-normalised

so the rest of the pipeline (structure head, A1/A2 losses, sanity report) is unchanged.
Attributes used by main.py: K, pad, per_channel, n_scales, name, summary().

    gabor   oriented band-pass Gabor energy (gabor.GaborBank, the Palm-JEPA default)
    sobel   oriented first-derivative energy |gx cos(t) + gy sin(t)| from 3x3 Sobel gradients
    hog     HOG cell histogram: [-1, 0, 1] gradients, magnitude-weighted, unsigned orientation,
            linear interpolation between the two nearest bins (no block normalisation; the
            per-patch L2 normalisation of the descriptor plays that role)
    ridge   Hessian crease response per orientation and scale: second derivative along direction t
            (across the line), kept where positive (dark line on bright skin), at the --ridge_sigmas scales
    lbp     rotation-invariant uniform LBP (8 neighbours, radius 1): one-hot code maps, so the
            patch average is the LBP histogram (texture, no orientation)
    random  fixed random filters with exactly the Gabor bank's shape (same kernel sizes, number of
            filters), zero-mean and unit-norm like the Gabor kernels; fixed seed, identical in
            every run -> controls for "any fixed filter bank" vs Gabor specifically

The number of orientations is --gabor_orient for gabor, sobel, hog and ridge.
--gabor_gray 0 computes every target per RGB channel (K x 3), as for Gabor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gabor import GaborBank
from line_masking import _gauss_deriv_kernels

STRUCT_TARGETS = ("gabor", "sobel", "hog", "ridge", "lbp", "random")


class _Base(nn.Module):
    """Shared per-channel / grayscale handling. Subclasses implement _single(x1) for (B,1,H,W)."""
    name = "base"
    k_single = 1

    def __init__(self, per_channel=False):
        super().__init__()
        self.per_channel = per_channel
        self.n_scales = 1
        self.pad = 1

    @property
    def K(self):
        return self.k_single * (3 if self.per_channel else 1)

    @torch.no_grad()
    def forward(self, x):
        x = x.float()
        if self.per_channel:
            B, C, H, W = x.shape
            r = self._single(x.reshape(B * C, 1, H, W))                    # (B*C, k, H, W)
            return r.reshape(B, C * r.size(1), H, W)                       # channel-major, like GaborBank
        return self._single(x.mean(dim=1, keepdim=True))

    def summary(self):
        return f"{self.name}: K={self.K} ({'per-channel RGB' if self.per_channel else 'grayscale'})"


class SobelTarget(_Base):
    name = "sobel"

    def __init__(self, n_orient=8, per_channel=False):
        super().__init__(per_channel)
        self.n_orient = self.k_single = n_orient
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("k", torch.stack([kx, kx.t()])[:, None])     # (2,1,3,3): gx, gy
        th = torch.arange(n_orient, dtype=torch.float32) * math.pi / n_orient
        self.register_buffer("cs", torch.stack([th.cos(), th.sin()], 1))  # (n_orient, 2)

    def _single(self, g):
        d = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), self.k)    # (N,2,H,W)
        return torch.einsum("oc,nchw->nohw", self.cs, d).abs()


class HOGTarget(_Base):
    name = "hog"

    def __init__(self, n_orient=8, per_channel=False):
        super().__init__(per_channel)
        self.n_orient = self.k_single = n_orient
        k = torch.tensor([-1., 0., 1.])
        self.register_buffer("kx", k.view(1, 1, 1, 3))
        self.register_buffer("ky", k.view(1, 1, 3, 1))

    def _single(self, g):
        gx = F.conv2d(F.pad(g, (1, 1, 0, 0), mode="replicate"), self.kx)
        gy = F.conv2d(F.pad(g, (0, 0, 1, 1), mode="replicate"), self.ky)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        ang = torch.atan2(gy, gx) % math.pi                               # unsigned orientation [0, pi)
        pos = ang / (math.pi / self.n_orient) - 0.5                       # bin centres at (b + 0.5) * width
        lo = torch.floor(pos)
        w_hi = pos - lo
        lo_i = lo.long() % self.n_orient
        hi_i = (lo_i + 1) % self.n_orient
        out = torch.zeros(g.size(0), self.n_orient, *g.shape[2:], device=g.device, dtype=g.dtype)
        out.scatter_add_(1, lo_i, mag * (1 - w_hi))
        out.scatter_add_(1, hi_i, mag * w_hi)
        return out


class RidgeTarget(_Base):
    name = "ridge"

    def __init__(self, n_orient=8, sigmas=(1.0, 1.5, 2.0), per_channel=False):
        super().__init__(per_channel)
        self.n_orient = n_orient
        self.sigmas = tuple(sigmas)
        self.n_scales = len(self.sigmas)
        self.k_single = n_orient * self.n_scales
        r_max = int(math.ceil(3 * max(self.sigmas)))
        ks = []
        for s in self.sigmas:                                             # pad every scale to the largest kernel
            k, r = _gauss_deriv_kernels(s, "cpu", torch.float32)
            ks.append(F.pad(k, (r_max - r,) * 4))
        self.register_buffer("k", torch.cat(ks, 0))                       # (3*S,1,k,k): Hxx, Hyy, Hxy per scale
        self.pad = r_max
        th = torch.arange(n_orient, dtype=torch.float32) * math.pi / n_orient
        # second derivative along direction (cos t, sin t), i.e. across a line whose normal has angle t
        # (same angle convention as the Gabor and Sobel targets):
        # D = Hxx cos^2 t + 2 Hxy sin t cos t + Hyy sin^2 t  (> 0 across a dark line)
        self.register_buffer("w", torch.stack([th.cos() ** 2, th.sin() ** 2,
                                               2 * th.sin() * th.cos()], 1))  # (n_orient, 3): Hxx, Hyy, Hxy

    def _single(self, g):
        h = F.conv2d(F.pad(g, (self.pad,) * 4, mode="replicate"), self.k)  # (N, 3*S, H, W)
        h = h.view(g.size(0), self.n_scales, 3, *g.shape[2:])
        d = torch.einsum("oc,nschw->nsohw", self.w, h).clamp_min(0)
        return d.reshape(g.size(0), self.k_single, *g.shape[2:])          # scale-major, like GaborBank


class LBPTarget(_Base):
    name = "lbp"
    # neighbours (dy, dx), in circular order
    OFFS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))

    def __init__(self, per_channel=False):
        super().__init__(per_channel)
        self.k_single = len(self.OFFS) + 2                                # riu2: 0..8 uniform, 9 = non-uniform

    def _single(self, g):
        H, W = g.shape[2:]
        p = F.pad(g, (1, 1, 1, 1), mode="replicate")
        bits = torch.stack([(p[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] >= g)
                            for dy, dx in self.OFFS], 0).float()          # (8, N, 1, H, W)
        ones = bits.sum(0)
        trans = (bits - bits.roll(1, dims=0)).abs().sum(0)                # 0/1 transitions around the circle
        code = torch.where(trans <= 2, ones, torch.full_like(ones, len(self.OFFS) + 1)).long()
        return F.one_hot(code[:, 0], self.k_single).permute(0, 3, 1, 2).to(g.dtype)


class RandomFilterBank(nn.Module):
    """Same shape and normalisation as GaborBank, random kernels (fixed seed)."""
    name = "random"

    def __init__(self, n_orient=8, scales=((5, 1.5, 3.0), (9, 3.0, 6.0), (13, 4.5, 9.0)),
                 per_channel=False, seed=0):
        super().__init__()
        self.n_orient = n_orient
        self.n_scales = len(scales)
        self.per_channel = per_channel
        n_filters = n_orient * self.n_scales
        self.K = n_filters * (3 if per_channel else 1)
        gen = torch.Generator().manual_seed(seed)
        max_k = max(s[0] for s in scales)
        bank = torch.zeros(n_filters, 1, max_k, max_k)
        idx = 0
        for (ksize, _, _) in scales:
            for _ in range(n_orient):
                k = torch.randn(ksize, ksize, generator=gen)
                k = k - k.mean()
                k = k / (k.norm() + 1e-8)
                pad = (max_k - ksize) // 2
                bank[idx, 0, pad:pad + ksize, pad:pad + ksize] = k
                idx += 1
        self.register_buffer("weight", bank)
        self.pad = max_k // 2

    @torch.no_grad()
    def forward(self, x):
        if self.per_channel:
            w = self.weight.repeat(3, 1, 1, 1)
            return F.conv2d(x, w, padding=self.pad, groups=3)
        return F.conv2d(x.mean(dim=1, keepdim=True), self.weight, padding=self.pad)

    def summary(self):
        return (f"random: K={self.K} ({self.n_orient} x {self.n_scales} fixed random zero-mean filters, "
                f"{'per-channel RGB' if self.per_channel else 'grayscale'})")


def build_struct_target(cfg):
    """The structure-target module selected by --struct_target."""
    kind = getattr(cfg, "struct_target", "gabor")
    per_channel = not bool(getattr(cfg, "gabor_gray", 1))
    if kind == "gabor":
        return GaborBank(n_orient=cfg.gabor_orient, scales=cfg.gabor_scales,
                         gamma=cfg.gabor_gamma, per_channel=per_channel)
    if kind == "sobel":
        return SobelTarget(cfg.gabor_orient, per_channel)
    if kind == "hog":
        return HOGTarget(cfg.gabor_orient, per_channel)
    if kind == "ridge":
        return RidgeTarget(cfg.gabor_orient, tuple(cfg.ridge_sigmas), per_channel)
    if kind == "lbp":
        return LBPTarget(per_channel)
    if kind == "random":
        return RandomFilterBank(cfg.gabor_orient, cfg.gabor_scales, per_channel)
    raise ValueError(f"unknown --struct_target {kind}")
