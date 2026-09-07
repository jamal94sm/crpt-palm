"""barlow_loss.py -- Barlow Twins loss, verified verbatim against
facebookresearch/barlowtwins/main.py (fetched 2026-09-03):

  c = BN(z1, affine=False).T @ BN(z2, affine=False)
  c /= batch_size
  loss = sum((diag(c) - 1)^2) + lambd * sum(off_diag(c)^2)

No stop-gradient anywhere -- both branches share one encoder+projector and
both contribute gradients to the same parameters (unlike SimSiam/BYOL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowBN(nn.Module):
    """The official code's self.bn = nn.BatchNorm1d(dim, affine=False) --
    normalizes each projector output dimension to zero-mean/unit-variance
    across the batch, purely for the loss computation (no learnable
    scale/shift, hence affine=False)."""

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False)

    def forward(self, x):
        return self.bn(x)


def barlow_loss(z1, z2, bn, lambd, batch_size):
    c = bn(z1).T @ bn(z2)
    c = c / batch_size

    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()
    loss = on_diag + lambd * off_diag

    with torch.no_grad():
        # Report std on the BN-NORMALIZED embeddings (what the loss itself
        # sees via bn(z1)/bn(z2)), not F.normalize(z1) -- these are
        # different operations (per-sample L2 norm vs. per-dim batch
        # statistics) and the mismatch was hiding what the loss actually
        # operates on.
        bn_z1, bn_z2 = bn(z1), bn(z2)
        stats = {"on_diag": on_diag.item(), "off_diag": off_diag.item(),
                  "std_z1": bn_z1.std(dim=0).mean().item(),
                  "std_z2": bn_z2.std(dim=0).mean().item(),
                  "off_diag_mean_abs": off_diagonal(c).abs().mean().item()}
    return loss, stats
