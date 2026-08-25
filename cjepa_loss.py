"""cjepa_loss.py -- C-JEPA regularizer (Mo & Tong, "Connecting Joint-Embedding
Predictive Architecture with Contrastive Self-supervised Learning", NeurIPS
2024, arXiv:2410.19560).

Reimplemented from the paper's Eq. 21-22 and Appendix C.2 Algorithm 1 -- no
official code is public for this paper (unlike VICReg itself), and the
paper's own two descriptions of this mechanism (the Python-ish pseudocode
vs. the formal Algorithm 1) aren't fully consistent with each other.
Algorithm 1 is treated as authoritative here since it's the shape-annotated,
precise version.

Mechanism: unlike VICReg, this needs NO separate augmented views and NO
extra encoder forward pass. It reuses I-JEPA's own M target-block predictor
outputs (already computed for the main JEPA loss) as the "multiple views":
for every ordered pair of target blocks (i, j), i != j, mean-pool each
block's predicted representation over patches, project through a small
shared head, and run the standard VICReg triple (sim + std + cov) on the
pair -- averaged over all M*(M-1) ordered pairs.

Total loss (paper Eq. 21): L = L_JEPA + cjepa_weight * L_reg
  where L_reg = sim_weight*sim + std_weight*std + cov_weight*cov   (Eq. 22)
  paper defaults: cjepa_weight=0.001, sim_weight=25, std_weight=25, cov_weight=1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CJEPAProjector(nn.Module):
    """3-layer BN-ReLU MLP, same skeleton as VICReg's own projector. The
    paper's Algorithm 1 just says "apply projection to the embeddings"
    without specifying architecture; this is a standard, reasonable default."""

    def __init__(self, in_dim, out_dim=None, hidden=None):
        super().__init__()
        hidden = hidden or in_dim
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _variance_term(z, gamma, eps):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def _covariance_term(z):
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    return _off_diagonal(cov).pow(2).sum() / D


def cjepa_regularizer(pred_embeds, num_blocks, batch_size, projector,
                       sim_weight=25.0, std_weight=25.0, cov_weight=1.0,
                       gamma=1.0, eps=1e-4):
    """
    pred_embeds : (num_blocks * batch_size, N_tgt, D) -- the predictor's
                  output already computed for the main JEPA loss, in
                  block-major row order (rows [0:B)=block 0, [B:2B)=block 1,
                  ...). Safe to assume here because ctx_masks always has
                  exactly 1 entry in this codebase (Patchify returns
                  [ctx_out]), which makes repeat_interleave_batch's
                  repeat=1 call a data-order no-op on top of apply_masks'
                  block-major torch.cat.
    num_blocks  : M, cfg.num_blocks.
    batch_size  : B.
    projector   : a CJEPAProjector instance.

    Returns (loss, stats). loss is UNWEIGHTED by cjepa_weight (Eq. 21's
    outer scale) -- apply that multiplier in the training loop, same
    pattern as w_a1/w_a2/w_sup.
    """
    B, M = batch_size, num_blocks
    if M < 2:
        raise ValueError("cjepa_regularizer needs num_blocks >= 2")
    D = pred_embeds.size(-1)

    pooled = pred_embeds.mean(dim=1)                 # (M*B, D), block-major
    pooled = pooled.view(M, B, D).permute(1, 0, 2)    # (B, M, D)

    zc = projector(pooled.reshape(B * M, D)).view(B, M, -1)  # (B, M, proj_dim)

    sim_sum = std_sum = cov_sum = 0.0
    n_pairs = 0
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            zci, zcj = zc[:, i, :], zc[:, j, :]
            sim_sum = sim_sum + F.mse_loss(zci, zcj)
            std_sum = std_sum + 0.5 * (_variance_term(zci, gamma, eps)
                                        + _variance_term(zcj, gamma, eps))
            cov_sum = cov_sum + _covariance_term(zci) + _covariance_term(zcj)
            n_pairs += 1

    sim_sum, std_sum, cov_sum = sim_sum / n_pairs, std_sum / n_pairs, cov_sum / n_pairs
    loss = sim_weight * sim_sum + std_weight * std_sum + cov_weight * cov_sum

    with torch.no_grad():
        stats = {"sim": float(sim_sum), "std": float(std_sum), "cov": float(cov_sum)}
    return loss, stats