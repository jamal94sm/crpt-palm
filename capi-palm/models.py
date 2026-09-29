"""models.py -- CAPI encoder/decoder, ported from the confirmed official
facebookresearch/capi model.py. Capacity SCALED DOWN to match this project's
convention (~4.91M params at embed_dim=256) rather than the paper's 300M
ViT-L -- encoder depth/heads follow this project's existing auto-derive rule
(same as ContextEncoder), decoder depth = encoder depth // 2 (paper's own
rule: "use a predictor depth equal to half that of the encoder").

Architecture is verbatim in STRUCTURE from the source: patch_embed -> drop to
visible_indices -> registers prepended -> encoder (self-attn only, RoPE) ->
decoder (cross-attn only against encoder's last-layer output, RoPE, no self-
attn among mask tokens -- confirmed by Block's context_dim wiring and
Transformer.forward's per-block `context` list). RMSNorm (paper's own choice,
Table 7) used throughout, matching source's default norm_layer_type.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope_sk import Attention


def _init_weights(m, xavier_gain=1):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=xavier_gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)) and getattr(m, "elementwise_affine", False):
        nn.init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super().__init__()
        hidden = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Residual(nn.Module):
    """Simplified from source's EfficientResidual: drop-path support kept,
    the batch-subsampling efficiency trick omitted (irrelevant at this
    project's tiny batch size; kept mathematically equivalent via the
    simpler NaiveResidual-style masking path from the source)."""

    def __init__(self, drop_prob, norm, fn):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.keep_prob = 1 - drop_prob

    def forward(self, x, **kwargs):
        fn_out = self.fn(self.norm(x), **kwargs)
        if self.keep_prob == 1.0 or not self.training:
            return x + fn_out
        mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[:, None, None]
        return x + fn_out * mask / self.keep_prob


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop_path, norm_layer, context_dim=None, mlp_ratio=4):
        super().__init__()
        self.residual1 = Residual(drop_path, norm_layer(dim),
                                  Attention(dim, num_heads, context_dim=context_dim))
        self.residual2 = Residual(drop_path, norm_layer(dim), Mlp(dim, mlp_ratio))

    def forward(self, x, context=None, coords=None, context_coords=None):
        x = self.residual1(x, context=context, coords=coords, context_coords=context_coords)
        return self.residual2(x)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, norm_layer, drop_path_rate=0.0, context_dim=None, mlp_ratio=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, drop_path_rate, norm_layer, context_dim, mlp_ratio)
            for _ in range(depth)])

    def forward(self, x, contexts=None, coords=None, context_coords=None):
        for i, blk in enumerate(self.blocks):
            context = contexts[i] if contexts is not None else None
            x = blk(x, context=context, coords=coords, context_coords=context_coords)
        return x


def _safe_heads(dim, ratio=32):
    h = max(1, dim // ratio)
    while dim % h != 0 and h > 1:
        h -= 1
    return h


class CapiEncoderDecoder(nn.Module):
    """encoder: self-attn only, sees only visible patches + registers.
    decoder: cross-attn only, mask tokens attend to encoder's final output;
    no self-attention among mask tokens (matches source: Block's residual1
    is Attention(context_dim=encoder.embed_dim), context passed identically
    to every decoder block, coords=predict-position coords throughout)."""

    def __init__(self, image_size, num_patches, embed_dim, depth=None, num_heads=None,
                 n_registers=4, drop_path_rate=0.0, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        self.patch_size = H // num_patches
        self.grid = num_patches
        num_heads = num_heads or _safe_heads(embed_dim)
        depth = depth or min(6, embed_dim // 64 + 2)
        dec_depth = max(1, depth // 2)             # paper's own rule: half the encoder depth

        # ADAPTED: decoder width decoupled from encoder width (this project's own
        # Predictor class does the same thing -- fixed internal pred_dim=128
        # regardless of embed_dim -- rather than the paper's width=embed_dim
        # decoder, to keep total baseline capacity comparable across this
        # project's ecosystem instead of ~1.5x larger than every sibling baseline).
        pred_dim = 128
        dec_heads = _safe_heads(pred_dim, ratio=32)
        norm_layer = lambda d: nn.RMSNorm(d, eps=1e-5)   # paper's Table 7 choice
        self.encoder = Transformer(embed_dim, depth, num_heads, norm_layer, drop_path_rate, mlp_ratio=mlp_ratio)
        self.dec_in_proj = nn.Linear(embed_dim, pred_dim)      # project context down to pred_dim
        self.decoder = Transformer(pred_dim, dec_depth, dec_heads, norm_layer, drop_path_rate,
                                   context_dim=pred_dim, mlp_ratio=mlp_ratio)
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim
        self.n_registers = n_registers

        self.mask_token = nn.Parameter(torch.empty(1, pred_dim))
        self.registers = nn.Parameter(torch.empty(1, n_registers, embed_dim))
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.enc_norm = norm_layer(embed_dim)
        self.dec_norm = norm_layer(pred_dim)
        self._init_weights()

    def _init_weights(self):
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.registers, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def _get_edge_coordinates(self, n, dtype, device):
        """Verbatim from source: registers placed evenly around the image
        border (n must be divisible by 4)."""
        side = n // 4
        assert n == 4 * side, "n_registers must be divisible by 4 (edge placement)"
        reg = torch.zeros(1, n, 2, dtype=dtype, device=device)
        c = torch.arange(side, dtype=dtype, device=device) / side
        reg[:, 0*side:1*side, 0] = c; reg[:, 0*side:1*side, 1] = 0
        reg[:, 1*side:2*side, 0] = 1; reg[:, 1*side:2*side, 1] = c
        reg[:, 2*side:3*side, 0] = 1 - c; reg[:, 2*side:3*side, 1] = 1
        reg[:, 3*side:4*side, 0] = 0; reg[:, 3*side:4*side, 1] = 1 - c
        return reg

    def prepare_tokens_and_drop(self, x, visible_indices):
        """x: (B,3,H,W). visible_indices: flat LongTensor indexing into the
        (B*grid*grid) flattened token space, or None (use all patches)."""
        b, _, h, w = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)             # (B, P, D)
        coord_x = torch.linspace(0, 1, h // self.patch_size, device=x.device, dtype=x.dtype)
        coord_y = torch.linspace(0, 1, w // self.patch_size, device=x.device, dtype=x.dtype)
        coords_all = torch.cartesian_prod(coord_x, coord_y)[None].expand(b, -1, -1)   # (B, P, 2)
        if visible_indices is not None:
            coords = coords_all.flatten(0, 1)[visible_indices].reshape(b, -1, 2)
            x = x.flatten(0, 1)[visible_indices].reshape(b, -1, self.embed_dim)
        else:
            coords = coords_all
        reg_coords = self._get_edge_coordinates(self.n_registers, x.dtype, x.device).expand(b, -1, -1)
        # scale registers to the visible-patch bounding box (source's scale_reg_to_visible=True)
        mi, ma = coords.min(dim=1, keepdim=True).values, coords.max(dim=1, keepdim=True).values
        reg_coords = mi + reg_coords * (ma - mi)
        x = torch.cat([self.registers.expand(b, -1, -1), x], dim=1)
        coords = torch.cat([reg_coords, coords], dim=1)
        return x, coords, coords_all

    def forward_pretrain(self, x, visible_indices=None, predict_indices=None, do_prediction=False):
        """Used at train time. Returns (encoder_patch_output (B,n_visible,D),
        decoder_output_flat (n_predict_total, D) or None)."""
        b, _, _, _ = x.shape
        z, coords_enc, coords_all = self.prepare_tokens_and_drop(x, visible_indices)
        enc_out = self.enc_norm(self.encoder(z, coords=coords_enc))
        dec_out = None
        if do_prediction:
            coords_dec = coords_all.flatten(0, 1)[predict_indices].reshape(b, -1, 2)
            n_pred = coords_dec.shape[1]
            mask_tok = self.mask_token[None].expand(b, n_pred, -1)
            enc_proj = self.dec_in_proj(enc_out)                          # (B, n_visible+reg, pred_dim)
            contexts = [enc_proj] * self.decoder.n_blocks
            dec_raw = self.decoder(mask_tok, contexts=contexts, coords=coords_dec, context_coords=coords_enc)
            dec_out = self.dec_norm(dec_raw).flatten(0, 1)               # (B*n_pred, pred_dim)
        return enc_out[:, self.n_registers:], dec_out

    def forward(self, x):
        """Used at eval time: full image, no masking, mean-pool patch
        tokens (matches this project's other baselines' pooling contract).
        """
        enc_out, _ = self.forward_pretrain(x, visible_indices=None, predict_indices=None, do_prediction=False)
        return enc_out.mean(dim=1)


class FeatureExtractor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)
