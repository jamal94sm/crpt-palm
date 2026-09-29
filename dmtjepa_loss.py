"""dmtjepa_loss.py -- Masked Semantic Neighboring + Local Aggregation Target
(Mo & Yun, arXiv:2405.17995), grafted onto this project's I-JEPA pipeline.

SCOPE NOTE (read before use): the paper's Local Aggregation Target module
has TWO symmetric heads -- a target-side head h_theta~ (Eq. 4, first line)
that builds the training TARGET from neighboring patches, and a context-
side head h_theta (Eq. 4, second line) whose output s_x^LAT is meant to
REPLACE the predictor's input. This project's Predictor class (models.py)
has a fixed interface that consumes ctx_embeds per-patch, with its own
positional-embedding and mask-token machinery built around that shape --
substituting a single pooled s_x^LAT vector in its place is a real
architecture change to Predictor, not a drop-in swap, and was NOT made
here. This integration therefore implements the TARGET-side aggregation
only (h_theta~ / target_agg_head): a discriminative, neighbor-aggregated
target for each masked patch, predicted by the EXISTING Predictor exactly
as it already predicts raw target-block representations in this project's
JEPA/C-JEPA baselines. The context-side head and its EMA counterpart are
NOT implemented, since including them without ever calling them (as an
earlier draft of this file did) would silently do nothing while still
consuming an EMA-update code path and optimizer slot for no effect.

Built directly against models.py's real tensor shapes:
    tgt_masks[k]:  (B, N_tgt) LongTensor of FLAT patch indices into an
                   H x W grid (H=W=num_patches), as produced by patchify()'s
                   idx = [(y+i)*W + (x+j) ...] -- so (row, col) recovers
                   exactly via idx // W, idx % W.
    tgt_full:      (B, P, D) target-encoder output BEFORE apply_masks() --
                   neighbor lookups must happen here, since apply_masks()
                   discards spatial layout by gathering into (B*n_masks, N, D).

target_agg_head (h_theta~): aggregates the k selected NEIGHBOR target
representations (Eq. 4, first line) into one discriminative target vector
per masked patch. EMA-updated from a SEPARATE, TRAINABLE context_agg_head
that participates in the loss (see source_pretraining.py's integration:
context_agg_head is called on ctx_embeds' mean-pooled representation only
to give it a supervised gradient signal via a small auxiliary consistency
term -- NOT to replace the predictor's input). If you do not want this
auxiliary term, target_agg_head can instead be updated by copying
context_agg_head's weights directly every N steps rather than via EMA;
see the --dmtjepa_head_update flag in config.py.

Cross-attention only (Table 8/9 in the paper: cross-attention beats both
average-pooling and self-attention decisively; this project does not
implement the pooling/self-attention ablation variants).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAggregationHead(nn.Module):
    """Single cross-attention layer (paper's ablated-best config: Table 8/9).
    query: (M, 1, D) one query per masked/context patch this head is called for.
    kv:    (M, k, D) the k neighbor / context vectors to aggregate over.
    -> (M, D) aggregated vector, via Eq. 5's softmax(q k^T / sqrt(D)) @ v.
    No separate value projection in the paper's formulation (Eq. 5 attends
    directly over {x_j} / s_x) -- this keeps the head "lightweight" as
    described in the abstract, and keeps its output in the same space as
    its own input (needed for the L2 loss against the predictor's output).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, query, kv):
        q = self.q_proj(query)                                   # (M, 1, D)
        k = self.k_proj(kv)                                      # (M, k, D)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.dim ** 0.5), dim=-1)
        return (attn @ kv).squeeze(1)                             # (M, D) -- values are RAW kv (Eq. 5)


def _neighbor_offsets(window):
    """window=3 -> the 8 offsets of a 3x3 neighborhood EXCLUDING the center
    (paper: 'a set of semantically similar NEIGHBORING patches'; the query
    patch itself is not its own neighbor). window must be odd."""
    assert window % 2 == 1, "neighbor window must be odd (e.g. 3 for 3x3)"
    r = window // 2
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if not (dy == 0 and dx == 0)]


def select_semantic_neighbors(tgt_full, tgt_idx_flat, grid, window, k):
    """Masked Semantic Neighboring (paper Eq. 3).

    tgt_full:      (B, P, D) target-encoder output, full grid.
    tgt_idx_flat:  (B, N_tgt) flat patch indices of the masked/query patches
                   (one target BLOCK's mask, i.e. one element of tgt_masks).
    grid:          H = W = num_patches (the patchify() grid size).
    window:        neighborhood size (odd, e.g. 3 for 3x3).
    k:             number of top-similarity neighbors to keep per query patch.

    Returns:
        neighbor_feats: (B, N_tgt, k_eff, D) the top-k_eff neighbor TARGET
                         features per query patch (paper's {x_j}_{j in P_i}).
                         k_eff = min(k, window^2 - 1): on this project's
                         small grid (e.g. 8x8), a 3x3 window has only 8
                         candidate neighbors even for interior patches, so
                         k should not exceed 8 when window=3.
        valid:          (B, N_tgt) bool -- False where fewer than k_eff
                         in-bounds neighbors existed (grid-edge patches).
                         Excluded from the loss by the caller: Eq. 3's
                         top-k ranking is undefined with under-k
                         candidates rather than silently padded.
    """
    B, N_tgt = tgt_idx_flat.shape
    D = tgt_full.size(-1)
    device = tgt_full.device

    rows = tgt_idx_flat // grid                                   # (B, N_tgt)
    cols = tgt_idx_flat % grid

    offsets = _neighbor_offsets(window)
    n_off = len(offsets)
    neigh_rows = rows.unsqueeze(-1) + torch.tensor([o[0] for o in offsets], device=device)
    neigh_cols = cols.unsqueeze(-1) + torch.tensor([o[1] for o in offsets], device=device)
    in_bounds = (neigh_rows >= 0) & (neigh_rows < grid) & (neigh_cols >= 0) & (neigh_cols < grid)
    neigh_flat = (neigh_rows.clamp(0, grid - 1) * grid + neigh_cols.clamp(0, grid - 1))

    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N_tgt, n_off)
    neigh_feats_all = tgt_full[batch_idx, neigh_flat]              # (B, N_tgt, n_off, D)

    query_feats = torch.gather(tgt_full, 1, tgt_idx_flat.unsqueeze(-1).expand(B, N_tgt, D))
    sim = F.cosine_similarity(query_feats.unsqueeze(2), neigh_feats_all, dim=-1)
    sim = sim.masked_fill(~in_bounds, float("-inf"))

    n_valid_per_query = in_bounds.sum(-1)
    k_eff = min(k, n_off)
    valid = n_valid_per_query >= k_eff
    top_sim, top_idx = sim.topk(k_eff, dim=-1)

    gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    neighbor_feats = torch.gather(neigh_feats_all, 2, gather_idx)
    return neighbor_feats, valid


def dmtjepa_targets(target_agg_head, tgt_full, tgt_masks, grid, window, k):
    """Builds the DMT-JEPA target for every target block in tgt_masks -- a
    drop-in replacement for apply_masks(tgt_full, tgt_masks) in
    source_pretraining.py's train_jepa. Call under torch.no_grad(), same as
    the existing tgt_full = target_encoder(images) line, since
    target_agg_head's parameters are not trained by backprop (EMA only).

    Returns:
        lat_targets: list of (B, N_tgt, D) tensors, one per target block
                     (matches apply_masks's original per-block shape, so
                     repeat_interleave_batch's call site is unchanged).
        valid_masks: list of (B, N_tgt) bool tensors, one per target block.
    """
    D = tgt_full.size(-1)
    lat_targets, valid_masks = [], []
    for m in tgt_masks:
        neighbor_feats, valid = select_semantic_neighbors(tgt_full, m, grid, window, k)
        B, N_tgt, k_eff, _ = neighbor_feats.shape
        query = torch.gather(tgt_full, 1, m.unsqueeze(-1).expand(B, N_tgt, D))
        q_flat = query.reshape(B * N_tgt, 1, D)
        kv_flat = neighbor_feats.reshape(B * N_tgt, k_eff, D)
        lat = target_agg_head(q_flat, kv_flat).view(B, N_tgt, D)
        lat_targets.append(lat)
        valid_masks.append(valid)
    return lat_targets, valid_masks


@torch.no_grad()
def update_ema_head(context_agg_head, target_agg_head, momentum):
    for pc, pt in zip(context_agg_head.parameters(), target_agg_head.parameters()):
        pt.data.mul_(momentum).add_(pc.data * (1.0 - momentum))


def context_consistency_loss(context_agg_head, ctx_embeds, lat_targets_detached):
    """Gives context_agg_head an actual training signal so its EMA copy
    (target_agg_head) is not just frozen at its random initialization
    forever. This is NOT part of the paper's Eq. 4/5 -- it is a minimal,
    clearly-labeled addition needed because this integration does not use
    the paper's s_x^LAT-replaces-predictor-input mechanism (see this
    file's module docstring). context_agg_head aggregates the VISIBLE
    context patches (using their own mean as the query, same shape
    convention as Eq. 4's x_c) and is pulled toward the mean of this
    step's already-computed PRE-REPEAT targets via a stop-gradient MSE --
    purely so it learns a sensible context-side aggregation function,
    analogous to the target head, rather than remaining at initialization.

    ctx_embeds:          (B, N_ctx, D) -- context_encoder's output, ONE
                          context mask (this project always uses exactly
                          one context mask per image; see patchify()).
    lat_targets_detached: list of (B, N_tgt, D) tensors, i.e. the
                          dmtjepa_targets() return value BEFORE
                          repeat_interleave_batch expands it across target
                          blocks -- using the pre-repeat list keeps the
                          batch dimension consistent with ctx_embeds' own
                          B, rather than B * n_target_blocks.

    If you do not want this term at all, pass --dmtjepa_ctx_weight 0
    (see config.py); target_agg_head then stays at its EMA-tracked
    initial weights, which does NOT match the paper's intent and is not
    recommended -- the flag exists for ablation purposes only.
    """
    query = ctx_embeds.mean(dim=1, keepdim=True)
    ctx_lat = context_agg_head(query, ctx_embeds)                        # (B, D)
    target_ref = torch.stack([t.mean(dim=1) for t in lat_targets_detached], dim=0).mean(dim=0)  # (B, D)
    return F.mse_loss(ctx_lat, target_ref)
