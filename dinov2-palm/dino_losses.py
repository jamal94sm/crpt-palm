"""dino_losses.py -- DINOLoss (CLS), iBOTPatchLoss, KoLeoLoss, single-GPU ports of dinov2/loss.

iBOTPatchLoss and KoLeoLoss follow the files read directly; dist.all_reduce, async centre
reduction and the static-shape padding to `upperbound` are dropped (single GPU, eager mode --
mathematically identical). DINOLoss mirrors the interface used in ssl_meta_arch.py; its body
is the standard DINO cross-entropy with EMA centring (dinov2/loss/dino_clstoken_loss.py itself
was not fetched). Sinkhorn-Knopp centring is intentionally NOT implemented: the official
default is plain centring and SK equipartitions over the *batch*, which is unreliable with
128 CLS rows spread over 2048 prototypes (batch 64).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                total_loss = total_loss - torch.sum(t * lsm, dim=-1).mean()
        return total_loss


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, patch_out_dim))

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        batch_center = teacher_patch_tokens.mean(dim=0, keepdim=True)   # mean over ALL masked tokens
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward_masked(self, student_masked, teacher_masked, masks_weight, n_images):
        """student_masked / teacher_masked: (n_masked, K) outputs at masked patches;
        masks_weight: (n_masked,) = 1 / (#masked patches of that image); n_images: rows of `masks`
        (official divides by masks.shape[0], i.e. images WITHOUT masked patches still count)."""
        if student_masked.shape[0] == 0:
            return student_masked.sum() * 0.0
        loss = torch.sum(teacher_masked * F.log_softmax(student_masked / self.student_temp, dim=-1), dim=-1)
        loss = loss * masks_weight
        return -loss.sum() / n_images


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic regulariser: spreads features by penalising small
    nearest-neighbour distances within a batch of L2-normalised CLS tokens."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def forward(self, student_output, eps=1e-8):
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        n = student_output.shape[0]
        dots = torch.mm(student_output, student_output.t())
        dots.view(-1)[:: (n + 1)].fill_(-1)                 # exclude self-similarity
        idx = torch.argmax(dots, dim=1)
        distances = self.pdist(student_output, student_output[idx])
        return -torch.log(distances + eps).mean()
