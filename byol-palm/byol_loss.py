"""byol_loss.py -- BYOL loss (Grill et al., NeurIPS 2020): normalized MSE
between the online predictor's output and the target network's projection,
symmetrized over both view assignments.

L = (2 - 2*cos_sim(p1, z2)) + (2 - 2*cos_sim(p2, z1))

Mathematically related to SimSiam's negative-cosine loss -- the difference
is architectural: BYOL's target branch is an EMA momentum copy of the FULL
online encoder+projector (computed under torch.no_grad() by the caller, so
no explicit .detach() is needed here), and BYOL's target has no predictor.
"""

import torch
import torch.nn.functional as F


def _neg_cosine_sim(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)


def byol_loss(p1, z2, p2, z1):
    l1 = _neg_cosine_sim(p1, z2)
    l2 = _neg_cosine_sim(p2, z1)
    loss = (l1 + l2).mean()

    with torch.no_grad():
        sim = 1.0 - loss.item() / 4.0
        std1 = F.normalize(z1, dim=-1).std(dim=0).mean().item()
        std2 = F.normalize(z2, dim=-1).std(dim=0).mean().item()
    return loss, {"sim": sim, "std_z1": std1, "std_z2": std2}