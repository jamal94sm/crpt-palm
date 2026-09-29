"""rope_sk.py -- RoPE attention and Sinkhorn-Knopp, ported verbatim from the
official facebookresearch/capi model.py (read directly, confirmed source).
Single-GPU: torch.distributed.all_reduce calls in stable_exp/reduced_sum are
dropped (no-op with world_size=1, mathematically identical).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope(nn.Module):
    """Axial RoPE: half the frequency channels encode x, half encode y.
    Verbatim from model.py's Rope class."""

    def __init__(self, dim, max_freq=7, min_freq=7e-4):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.freqs = nn.Parameter(torch.empty(2, self.dim))
        self._device_weight_init()

    def _device_weight_init(self):
        freqs_1d = self.max_freq * (self.max_freq / self.min_freq) ** torch.linspace(0, -1, self.dim // 4)
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2:] = freqs_1d
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x, coords):
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


class Attention(nn.Module):
    """Verbatim from model.py's Attention: self- or cross-attention depending
    on whether `context` is passed, both paths RoPE'd with their own coords."""

    def __init__(self, dim, num_heads, qkv_bias=False, proj_bias=False, context_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        context_dim = context_dim or dim
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = Rope(dim=head_dim)

    def forward(self, x, coords, context=None, context_coords=None):
        if context is None or context_coords is None:
            context, context_coords = x, coords
        b, n_q, d = x.shape
        b, n_k, _ = context.shape
        h = self.num_heads
        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        q = self.rope(q, coords[:, None, :, :])
        k = self.rope(k, context_coords[:, None, :, :])
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape([b, n_q, d])
        return self.proj(x)


exp_max_values = {torch.float16: 0, torch.float32: 50, torch.float64: 50, torch.bfloat16: 50}


def stable_exp(M):
    shift = M.max(dim=-2, keepdim=True).values
    M = M + (exp_max_values[M.dtype] - shift)
    return M.exp()


@torch.no_grad()
def sinkhorn_knopp(M, n_iterations, eps=1e-8):
    """Verbatim from model.py. M's dim=-2 is the axis normalized to
    near-uniform marginals (batch axis when positionwise; batch*position
    flattened otherwise -- see OnlineClustering's transpose/flatten choice)."""
    M = stable_exp(M)
    for _ in range(n_iterations):
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
    return M


class OnlineClustering(nn.Module):
    """Verbatim from model.py: no weight-norm (unlike L2NormLinear), has its
    OWN loss (Eq. 3's minimum-entropy clustering objective, cross-entropy
    between logits and their own SK-rebalanced assignments)."""

    def __init__(self, in_dim, out_dim, n_sk_iter, target_temp, pred_temp, bias=False, positionwise_sk=True):
        super().__init__()
        self.out_dim = out_dim
        self.n_sk_iter = n_sk_iter
        self.target_temp = target_temp
        self.pred_temp = pred_temp
        self.positionwise_sk = positionwise_sk
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.normal_(self.layer.weight, std=1)
        if bias:
            torch.nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x_n = F.normalize(x, dim=-1, p=2, eps=1e-7)
        logits = self.layer(x_n)
        sk_in = logits if self.positionwise_sk else logits.flatten(0, -2)
        assignments = sinkhorn_knopp(sk_in.detach() / self.target_temp, n_iterations=self.n_sk_iter)
        if not self.positionwise_sk:
            assignments = assignments.unflatten(0, logits.shape[:-1])
        tgt = assignments.flatten(0, -2).float()
        pred = logits.flatten(0, -2).float()
        loss = -torch.sum(tgt * F.log_softmax(pred / self.pred_temp, dim=-1), dim=-1).mean()
        return assignments.detach(), loss


class L2NormLinear(nn.Module):
    """Verbatim from model.py: L2-normalize, then a WEIGHT-NORMALIZED linear
    layer (weight_g fixed to 1, unlike OnlineClustering's plain linear)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        torch.nn.init.trunc_normal_(self.last_layer.parametrizations.weight.original1, std=0.02)
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, eps=eps)
        return self.last_layer(x)
