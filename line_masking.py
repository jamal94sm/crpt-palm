"""line_masking.py -- OPTIONAL line-guided target masking for SA-JEPA.

Not from the literature: the sampling distribution below is this project's
own design. Target-block CENTERS are drawn from

    p(i) = (1 - eps) * softmax(s / tau)_i + eps / P

where s_i is a per-patch line-saliency score computed from the RAW (not
descriptor-normalised) Gabor responses of the clean image. Block size, shape
(aspect ratio) and count are exactly I-JEPA's; the context-block sampling is
copied unchanged from models.patchify. eps = 1 recovers uniform placement.

Border handling: GaborBank zero-pads, so responses within `bank.pad` pixels of
the image edge are contaminated (the image/zero step looks like a line).
Reflect-padding (padded_responses) fixes that but creates a mirror "fold" that
also looks like a line when the image has a brightness gradient at the edge
(vignetting, common on smartphones). line_saliency(..., border=bank.pad)
instead ignores that band when pooling: each patch is scored only on pixels
whose filter response never touched the padding.

models.patchify is NOT modified, so JEPA / C-JEPA / DMT-JEPA are unaffected.
Return signature is identical to models.patchify: ([ctx (B,Nc)], [tgt_k (B,Nt)]).
"""
import math
import torch
import torch.nn.functional as F


def padded_responses(gabor_bank, images, pad=None):
    """Reflect-pad -> bank -> crop. Kept for comparison in check_line_saliency.py.
    Prefer line_saliency(gabor_bank(images), grid, border=gabor_bank.pad)."""
    if pad is None:
        pad = int(getattr(gabor_bank, "pad", 16))
    H, W = images.shape[-2:]
    if pad >= min(H, W):
        raise ValueError(f"reflect padding ({pad}) must be smaller than the image size ({H}x{W}).")
    x = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    r = gabor_bank(x)
    if r.dim() != 4 or tuple(r.shape[-2:]) != (H + 2 * pad, W + 2 * pad):
        raise ValueError(
            f"padded_responses needs a size-preserving bank; got {tuple(r.shape)} "
            f"for a {tuple(x.shape)} input.")
    return r[..., pad:pad + H, pad:pad + W]


def line_saliency(gabor_resp, grid, border=0, use_selectivity=False):
    """gabor_resp: raw Gabor responses (B, K, H, W) straight from gabor_bank(images).
    Returns per-patch saliency (B, grid*grid), z-scored within each image,
    in the SAME row-major patch order as the encoder (index = row*grid + col).

    border: pixels at the image edge to ignore when pooling. Use gabor_bank.pad
            (the kernel radius) to drop exactly the zero-padding-contaminated band.
            0 = pool over all pixels (old behaviour).
    use_selectivity: False (default) -> s = sum_k e_k (line energy).
            True -> s = energy * (max_k e_k / mean_k e_k). The selectivity factor
            is biased upward on border patches (fewer valid pixels -> noisier max),
            so it is off by default."""
    if gabor_resp.dim() != 4:
        raise ValueError(
            f"line_saliency expects raw Gabor responses of shape (B, K, H, W), "
            f"got {tuple(gabor_resp.shape)}. Check what gabor_bank(images) returns.")
    a = gabor_resp.abs().float()
    H, W = a.shape[-2:]
    if border > 0:
        if border >= min(H, W) // grid:
            raise ValueError(f"border ({border}px) must be smaller than the patch size "
                             f"({min(H, W) // grid}px), or corner patches have no valid pixels.")
        m = torch.zeros(1, 1, H, W, device=a.device, dtype=a.dtype)
        m[..., border:H - border, border:W - border] = 1.0
        e = F.adaptive_avg_pool2d(a * m, grid) / F.adaptive_avg_pool2d(m, grid).clamp_min(1e-8)
    else:
        e = F.adaptive_avg_pool2d(a, grid)                         # (B, K, g, g)
    e = e.flatten(2).transpose(1, 2)                               # (B, P, K), row-major patches
    s = e.sum(-1)
    if use_selectivity:
        s = s * (e.max(-1).values / (e.mean(-1) + 1e-6))
    return (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)


def patchify_line_guided(saliency, batch_size, num_patches, num_blocks=2,
                         trg_ratio=(0.10, 0.15), ctx_ratio=(0.90, 1.00),
                         ar_range=(0.75, 1.5), eps=0.5, tau=1.0, device="cpu"):
    """Same as models.patchify, except each TARGET block's position is centred
    on a patch drawn from p(i) above. saliency: (batch_size, P)."""
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    if not tau > 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    H = W = num_patches
    P = H * W
    if tuple(saliency.shape) != (batch_size, P):
        raise ValueError(f"saliency must be ({batch_size}, {P}), got {tuple(saliency.shape)}")

    probs_all = ((1.0 - eps) * torch.softmax(saliency.detach().float() / tau, dim=1)
                 + eps / P).cpu()

    def block_hw(scale):                                   # identical to patchify
        s = torch.empty(()).uniform_(*scale).item()
        ar = torch.empty(()).uniform_(*ar_range).item()
        area = max(1, int(s * P))
        h = max(1, min(H, int(round(math.sqrt(area * ar)))))
        w = max(1, min(W, int(round(area / h))))
        return h, w

    def sample_block_uniform(scale):                       # identical to patchify (used for context)
        h, w = block_hw(scale)
        y = torch.randint(0, max(1, H - h + 1), ())
        x = torch.randint(0, max(1, W - w + 1), ())
        idx = [(y + i) * W + (x + j) for i in range(h) for j in range(w)]
        return torch.tensor(idx, device=device)

    def sample_block_guided(scale, probs):                 # the only new sampling rule
        h, w = block_hw(scale)
        c = int(torch.multinomial(probs, 1))
        cy, cx = divmod(c, W)
        y = min(max(cy - h // 2, 0), H - h)
        x = min(max(cx - w // 2, 0), W - w)
        idx = [(y + i) * W + (x + j) for i in range(h) for j in range(w)]
        return torch.tensor(idx, device=device)

    ctx_masks, tgt_masks = [], [[] for _ in range(num_blocks)]
    min_ctx, min_tgt = P, P
    for b in range(batch_size):
        occupied = torch.zeros(P, dtype=torch.bool, device=device)
        for k in range(num_blocks):
            idx = sample_block_guided(trg_ratio, probs_all[b])
            tgt_masks[k].append(idx)
            occupied[idx] = True
            min_tgt = min(min_tgt, idx.numel())
        for _ in range(10):
            ctx = sample_block_uniform(ctx_ratio)
            ctx = ctx[~occupied[ctx]]
            if ctx.numel() > 0:
                break
        else:
            ctx = (~occupied).nonzero().squeeze(1)
        min_ctx = min(min_ctx, ctx.numel())
        ctx_masks.append(ctx)
    ctx_out = torch.stack([c[torch.randperm(c.numel(), device=device)[:min_ctx]] for c in ctx_masks])
    tgt_out = [torch.stack([t[torch.randperm(t.numel(), device=device)[:min_tgt]] for t in tgt_masks[k]])
               for k in range(num_blocks)]
    return [ctx_out], tgt_out


@torch.no_grad()
def target_topq_fraction(saliency, tgt_masks, q=0.25):
    """Diagnostic: fraction of target patches lying in each image's top-q
    saliency patches. Compare against a mask_mode=random run."""
    B, P = saliency.shape
    k = max(1, int(round(q * P)))
    top = torch.zeros(B, P, dtype=torch.bool, device=saliency.device)
    top.scatter_(1, saliency.topk(k, dim=1).indices, True)
    hits = [torch.gather(top, 1, m.to(saliency.device)).float().mean().item() for m in tgt_masks]
    return sum(hits) / len(hits)
