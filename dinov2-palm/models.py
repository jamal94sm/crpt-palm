"""models.py -- DINOv2 student/teacher ViT + projection head.

Capacity matched to this project's ContextEncoder/PlainViT (same depth/heads rule, 4x MLP,
pre-norm, fixed 2-D sin-cos position embedding, ~4.91M params at embed_dim=256), with the
DINOv2 ingredients: CLS token, learnable mask token, LayerScale, uniform stochastic depth.

CRITICAL ORDER (verified against dinov2/models/vision_transformer.py, prepare_tokens_with_masks):
    patch_embed -> replace masked patch embeddings by mask_token -> prepend CLS -> ADD pos-embed.
Position information must be added AFTER the substitution; otherwise every masked token is an
identical vector and the (permutation-equivariant) transformer cannot tell masked positions apart.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    assert embed_dim % 4 == 0, f"embed_dim={embed_dim} must be divisible by 4 (2-D sin-cos pos-embed)"

    def get_1d(dim, pos):
        omega = np.arange(dim // 2, dtype=float) / (dim / 2.0)
        omega = 1.0 / (10000 ** omega)
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    gh = np.arange(grid_size, dtype=float)
    gw = np.arange(grid_size, dtype=float)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape([2, 1, grid_size, grid_size])
    return np.concatenate([get_1d(embed_dim // 2, grid[0]), get_1d(embed_dim // 2, grid[1])], axis=1)


def _safe_heads(dim, target_ratio=32):
    heads = max(1, dim // target_ratio)
    while dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    def __init__(self, dim, init_values):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path, layerscale_init):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, layerscale_init) if layerscale_init > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, layerscale_init) if layerscale_init > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class DinoV2ViT(nn.Module):
    def __init__(self, image_size, num_patches, embed_dim, depth=None, num_heads=None,
                 mlp_ratio=4.0, drop_path_rate=0.0, layerscale_init=1e-5):
        super().__init__()
        assert image_size % num_patches == 0, "img_size must be divisible by num_patches"
        patch = image_size // num_patches
        heads = num_heads or _safe_heads(embed_dim, 32)
        depth = depth or min(6, embed_dim // 64 + 2)
        assert embed_dim % heads == 0, f"embed_dim={embed_dim} not divisible by heads={heads}"

        self.grid_size = num_patches
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))          # official: zeros
        pos = torch.tensor(get_2d_sincos_pos_embed(embed_dim, num_patches)).float().unsqueeze(0)
        self.pos_embed = nn.Parameter(torch.cat([torch.zeros(1, 1, embed_dim), pos], dim=1),
                                      requires_grad=False)                 # fixed, like every baseline here
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio, drop_path_rate, layerscale_init)
                                     for _ in range(depth)])                # uniform drop-path (official)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pos(self, n_patches):
        """Position embedding for a crop with n_patches tokens (local crops are smaller)."""
        if n_patches == self.grid_size ** 2:
            return self.pos_embed
        g = int(round(math.sqrt(n_patches)))
        assert g * g == n_patches, f"non-square token grid ({n_patches} tokens)"
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(1, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(g, g), mode="bicubic", align_corners=False)
        return torch.cat([cls_pos, patch_pos.permute(0, 2, 3, 1).reshape(1, g * g, -1)], dim=1)

    def forward_features(self, x, masks=None):
        """x: (B,3,H,W); masks: (B, n_patches) bool (True = masked). Returns dict(cls, patch)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)                  # (B, N, D)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self._pos(x.shape[1] - 1)                                   # AFTER mask substitution
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {"cls": x[:, 0], "patch": x[:, 1:]}


class DINOHead(nn.Module):
    """DINOv2 head: MLP -> L2-normalise -> weight-normalised linear to the prototypes
    (no norm_last_layer switch: weight_g starts at 1 and stays trainable, as in dinov2)."""

    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            layers = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1, p=2, eps=1e-12)
        return self.last_layer(x)


class FeatureExtractor(nn.Module):
    """forward(x) -> (B, embed_dim): CLS token (official eval) or mean of patch tokens."""

    def __init__(self, encoder, use_cls=True):
        super().__init__()
        self.encoder = encoder
        self.use_cls = use_cls
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            out = self.encoder.forward_features(x)
            return out["cls"] if self.use_cls else out["patch"].mean(dim=1)
