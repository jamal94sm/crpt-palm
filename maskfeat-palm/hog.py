"""hog.py -- HOG target-feature generator for MaskFeat.

Verified, faithful port of mmpretrain's HOGGenerator
(open-mmlab/mmpretrain/models/selfsup/maskfeat.py, fetched directly).
Fixed (non-learned) Sobel gradients -> Gaussian-weighted soft orientation
binning -> per-cell L2 normalization (the "local contrast normalization"
the MaskFeat paper identifies as essential -- Table 8a: -1.4% without it).

pool=7 here (not official's pool=8) because this project's patch size is
14px (img_size=112, num_patches=8 grid), not the official 16px -- pool=7
divides 14 evenly, giving exactly 2x2 HOG cells per patch, the same
cells-per-patch ratio the official 8px-pool/16px-patch config uses.
Output dim per patch = nbins(9) x cells(4) x channels(3) = 108,
coincidentally identical to the official config's 108.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HOGGenerator(nn.Module):
    def __init__(self, nbins=9, pool=7, gaussian_window=14):
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).contiguous()
        weight_y = weight_x.transpose(2, 3).contiguous()
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gaussian_kernel = self._get_gaussian_kernel(gaussian_window, gaussian_window // 2)
            self.register_buffer("gaussian_kernel", gaussian_kernel)

    @staticmethod
    def _get_gaussian_kernel(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= max(std, 1e-6)
        w = torch.exp(-0.5 * n**2)
        kernel_1d = w
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d / kernel_2d.sum()

    def _reshape_to_patches(self, hog_feat, grid_size):
        """hog_feat: (B, C, nbins, Hc, Wc) where Hc/Wc are HOG-cell counts.
        Reshape to (B, num_patches, nbins*cells_per_patch*C)."""
        B, C, nbins, Hc, Wc = hog_feat.shape
        cells_per_side = Hc // grid_size
        hog_feat = hog_feat.view(B, C * nbins, Hc, Wc)
        hog_feat = hog_feat.unfold(2, cells_per_side, cells_per_side)
        hog_feat = hog_feat.unfold(3, cells_per_side, cells_per_side)
        # (B, C*nbins, grid, grid, cells_per_side, cells_per_side)
        hog_feat = hog_feat.permute(0, 2, 3, 1, 4, 5).contiguous()
        hog_feat = hog_feat.view(B, grid_size * grid_size, -1)
        return hog_feat

    @torch.no_grad()
    def forward(self, x, grid_size):
        """x: (B, 3, H, W) clean image. grid_size: patch grid (e.g. 8).
        Returns: (B, num_patches, out_dim) target HOG features."""
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm = torch.stack([gx, gy], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx, gy) / self.pi * self.nbins

        b, c, h, w = norm.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm = norm.view(b, c, 1, h, w)

        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, f"h={h} gw={self.gaussian_window}"
                repeat_rate = h // self.gaussian_window
                gk = self.gaussian_kernel.repeat([repeat_rate, repeat_rate])
            else:
                gk = self.gaussian_kernel
            norm = norm * gk

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])
        out = F.normalize(out, p=2, dim=2)   # local contrast normalization

        return self._reshape_to_patches(out, grid_size)