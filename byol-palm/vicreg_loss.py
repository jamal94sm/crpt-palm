"""vicreg_loss.py -- VICReg loss (Bardes, Ponce, LeCun 2022):
invariance + variance + covariance, computed over two augmented views of
the same batch. Standard paper formula, paper-default weights (25/25/1)."""

import torch
import torch.nn.functional as F


def off_diagonal(x):
    """All off-diagonal elements of a square (D,D) matrix, flattened."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_loss(z, gamma=1.0, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    return off_diagonal(cov).pow(2).sum() / D


def vicreg_loss(z1, z2, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
                 gamma=1.0, eps=1e-4):
    """z1, z2: (B, D) projector outputs for the two views. Returns (loss, stats)."""
    inv = F.mse_loss(z1, z2)
    var = 0.5 * (variance_loss(z1, gamma, eps) + variance_loss(z2, gamma, eps))
    cov = covariance_loss(z1) + covariance_loss(z2)

    loss = lambda_inv * inv + lambda_var * var + lambda_cov * cov

    with torch.no_grad():
        stats = {
            "inv": inv.item(),
            "var": var.item(),
            "cov": cov.item(),
            "std1": z1.std(dim=0).mean().item(),
            "std2": z2.std(dim=0).mean().item(),
        }
    return loss, stats