"""simsiam_loss.py -- SimSiam loss (Chen & He, CVPR 2021): symmetrized
negative cosine similarity with a stop-gradient on the target branch.

L = 0.5 * D(p1, stopgrad(z2)) + 0.5 * D(p2, stopgrad(z1))
D(p, z) = -(p / ||p||_2) . (z / ||z||_2)

No negative pairs, no momentum/EMA encoder, no variance/covariance
regularization -- the paper's own analysis (Sec. 5) attributes collapse
prevention entirely to the predictor + stop-gradient asymmetry.
"""

import torch
import torch.nn.functional as F


def _neg_cosine_sim(p, z):
    """z is detached (stop-gradient) inside this function -- caller does
    not need to call .detach() itself."""
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


def simsiam_loss(p1, z1, p2, z2):
    """p1, p2: predictor h() outputs for view1/view2.
    z1, z2: projector g() outputs for view1/view2 (pre-predictor).
    Returns (loss, stats)."""
    loss = 0.5 * _neg_cosine_sim(p1, z2) + 0.5 * _neg_cosine_sim(p2, z1)

    with torch.no_grad():
        sim = -loss.item()   # positive cosine similarity, for logging
        # Per-dim std of L2-normalized z -- SimSiam's own collapse
        # diagnostic (paper Fig. 2): should stay well above 0, not
        # crash toward it.
        std1 = F.normalize(z1, dim=-1).std(dim=0).mean().item()
        std2 = F.normalize(z2, dim=-1).std(dim=0).mean().item()
    return loss, {"sim": sim, "std_z1": std1, "std_z2": std2}