"""sigreg_loss.py -- Sketched Isotropic Gaussian Regularization (Balestriero & LeCun,
arXiv:2511.08544), single-GPU port of Algorithm 1. galilai-group/lejepa is a
standalone loss-only library ("core SIGReg loss can be integrated into any
pretraining codebase") -- confirmed via its own README; no model/training code
exists there to port, only this loss. This file ports Algorithm 1 (Epps-Pulley
SIGReg) verbatim, dropping only the DDP all_reduce (single-GPU here).
"""
import math
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Epps-Pulley SIGReg (paper's recommended test, Section 4.2.3).
    x: (N, K) embeddings. num_slices = |A| (paper default 1024 for ViT-L/inet1k;
    ADAPTED to 256 here as a starting point for this project's much smaller
    batch size and embedding dim -- see config.py's --lejepa_num_slices help).
    integration domain [-5,5] and num_points=17 are the paper's own recommended
    defaults (Table 1a: negligible effect on performance)."""

    def __init__(self, num_slices=256, integration_bound=5.0, num_points=17):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(-integration_bound, integration_bound, num_points)
        self.register_buffer("t", t)
        self.register_buffer("exp_f", torch.exp(-0.5 * t ** 2))   # theoretical CF of N(0,1)

    def forward(self, x, generator=None):
        """x: (N, K). generator: optional torch.Generator for reproducible slice
        sampling (paper resamples A every step -- pass a fresh seed per call,
        e.g. derived from global_step, for that behavior)."""
        N, K = x.shape
        dev = x.device
        A = torch.randn(K, self.num_slices, device=dev, generator=generator)
        A = A / A.norm(p=2, dim=0, keepdim=True)                  # unit-norm directions

        x_t = (x @ A).unsqueeze(-1) * self.t                      # (N, M, T)
        ecf = torch.exp(1j * x_t).mean(dim=0)                     # (M, T) empirical CF

        err = (ecf - self.exp_f).abs().square() * self.exp_f      # (M, T)
        T_stat = torch.trapz(err, self.t, dim=-1) * N              # (M,) per-slice Epps-Pulley
        return T_stat.mean()                                       # scalar, averaged over slices (SIGReg def.)


def lejepa_prediction_loss(embeddings, n_global, n_views, batch_size):
    """Eq. 7's prediction loss: every view is pulled toward the mean of the
    global views, no predictor network needed.
    embeddings: (n_views * B, K), VIEW-MAJOR order (view v's B embeddings occupy
    rows [v*B:(v+1)*B) -- matches this project's multicrop_dataset collate,
    which stacks per-crop batches, and repeat_interleave_batch's own convention
    is NOT used here since there is no predictor/target-block structure).
    n_global: number of global views (first n_global views by convention).
    Returns: scalar loss, and (B, K) the global-view mean (mu_n in the paper,
    exposed for reuse -- SIGReg is applied per-view separately by the caller,
    not to this mean).
    """
    K = embeddings.size(-1)
    z = embeddings.view(n_views, batch_size, K)                    # (V, B, K)
    mu = z[:n_global].mean(dim=0)                                  # (B, K), paper's mu_n
    loss = ((mu.unsqueeze(0) - z) ** 2).sum(-1).mean()             # Eq. 7, mean over V and B
    return loss, mu
