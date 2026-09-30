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




# ══════════════════════════════════════════════════════════════
#  DMTPredictor -- faithful architecture (paper Section 2.3, confirmed
#  from arxiv.org/abs/2405.17995's own text + DMTJEPA/DMTJEPA repo README):
#  "the predictor g_theta(.,.) takes as input the output of the context
#  patch aggregation head s_x^a and a mask token for each patch to predict
#  {m_j}_{j in B_i}" -- s_x^LAT is a SINGLE pooled vector (paper: "x_c
#  denotes the AVERAGED embeddings ... only unmasked patches in the
#  context encoder"), not a per-patch sequence. This is why DMT-JEPA
#  cannot reuse this project's shared Predictor class (whose context
#  argument must be a per-patch sequence aligned with context_masks) --
#  a SEPARATE predictor class is the faithful integration, not a
#  modification of Predictor (which JEPA/C-JEPA/SA-JEPA all still share
#  unchanged).
# ══════════════════════════════════════════════════════════════
import torch.nn as _nn
from models import _gather


class DMTPredictor(_nn.Module):
    """Single-pooled-context predictor, faithful to DMT-JEPA's real
    architecture. context_agg_head is called INSIDE this module's forward
    (not as a side computation with its own loss) so its gradient comes
    entirely from the main JEPA loss backpropagating through this
    predictor -- exactly matching the paper, and structurally impossible
    to have the gradient-leak bug the earlier auxiliary-loss version had,
    since there is no separate loss term at all.

    Capacity matches this project's Predictor convention: pred_dim=128,
    depth=6, num_heads=2, fixed 2-D sin-cos position embedding for target
    positions (context has no position embedding here, since it's a
    single pooled vector with no spatial index of its own).
    """

    def __init__(self, num_patches, embed_dim, context_agg_head, norm_struct_out=True):
        super().__init__()
        pred_dim = 128
        depth = 6
        num_heads = 2

        self.context_agg_head = context_agg_head        # h_theta, trained via THIS predictor's loss only
        self.ctx_in_proj = _nn.Linear(embed_dim, pred_dim)   # projects s_x^LAT (B, D) -> (B, pred_dim)
        self.mask_token = _nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.out_proj = _nn.Linear(pred_dim, embed_dim)

        pos = get_2d_sincos_pos_embed_(pred_dim, num_patches)
        self.pos_embed = _nn.Parameter(torch.tensor(pos).float().unsqueeze(0), requires_grad=False)

        enc = torch.nn.TransformerEncoderLayer(d_model=pred_dim, nhead=num_heads,
                                               dim_feedforward=pred_dim * 4,
                                               batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(enc, depth)
        self.norm = _nn.LayerNorm(pred_dim)

    def forward(self, ctx_embeds, target_masks):
        """ctx_embeds: (B, N_ctx, D) -- the SAME visible-patch context
        embeddings this project's other JEPA baselines already compute
        (context_agg_head pools them internally into s_x^LAT here, this
        is NOT done by the caller). target_masks: list of (B, N_tgt)
        flat patch-index tensors, same convention as patchify()'s output.

        Returns: list of (B, N_tgt, D) predictions, one per target block
        -- same per-block list shape the caller already expects from the
        shared Predictor, so train_jepa's downstream code (repeat_
        interleave_batch, loss computation) needs minimal changes.
        """
        B = ctx_embeds.size(0)
        query = ctx_embeds.mean(dim=1, keepdim=True)             # (B, 1, D) -- paper's x_c
        s_x_lat = self.context_agg_head(query, ctx_embeds)        # (B, D) -- paper's s_x^LAT, GRADIENT FLOWS THROUGH
        ctx_tok = self.ctx_in_proj(s_x_lat).unsqueeze(1)          # (B, 1, pred_dim)

        preds = []
        for m in target_masks:
            N_tgt = m.size(1)
            pos_tgt = _gather(self.pos_embed.expand(B, -1, -1), m)   # (B, N_tgt, pred_dim)
            mask_tok = self.mask_token.expand(B, N_tgt, -1) + pos_tgt
            x = torch.cat([ctx_tok, mask_tok], dim=1)                 # (B, 1+N_tgt, pred_dim)
            x = self.norm(self.encoder(x))
            preds.append(self.out_proj(x[:, 1:]))                     # (B, N_tgt, D) -- drop the ctx token
        return preds


def get_2d_sincos_pos_embed_(embed_dim, grid_size):
    """Local alias so this file has no import-order dependency on models.py
    at module-load time (avoids a circular import if dmtjepa_loss.py is
    ever imported before models.py in some call site)."""
    from models import get_2d_sincos_pos_embed
    return get_2d_sincos_pos_embed(embed_dim, grid_size)
