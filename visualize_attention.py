"""visualize_attention.py -- attention map of a trained JEPA-family ContextEncoder on palm ROI images.

WHAT IS SHOWN
The encoder has no CLS token and the evaluation embedding is the MEAN of the patch
tokens (FeatureExtractor). The map therefore shows how much each INPUT patch
contributes to that embedding, via attention rollout (Abnar & Zuidema, 2020):
    per layer   A_l = mean over heads of the attention matrix            (N x N)
                A^_l = row-normalise(0.5*A_l + 0.5*I)                    (residual connection)
    rollout     R = A^_L ... A^_1                                        (R[i,j]: share of output token i
                                                                          that comes from input patch j)
    map         c_j = (1/N) * sum_i R[i,j]                               (mean pooling = average over i)
c sums to 1 over the N patches. Reshaped row-major to (grid, grid).

SECOND ROW: PCA map of the final patch features (what the features encode, not where information flows from)
    Z = final patch tokens of ALL images stacked (B*N x D); centre; top-3 principal components via SVD;
    scores P = (Z - mean) @ V[:3]^T; each score is clipped to its 2-98 percentile and mapped to R, G, B.
    The PCA is fitted JOINTLY on all images, so one colour means the same kind of feature in every image.
    Colours are arbitrary per model (only the grouping of patches can be compared, not the hue).
    Printed with it: variance explained per component, and each component's POSITIONAL SHARE (how much of its
    variance is the same in every image at a given patch position; ~1/B is chance, ~1 means the component just
    encodes location and not image content).

Run from the repo root (models.py importable):
  python visualize_attention.py --ckpt out_domain_analysis_xjtu/JEPA_context_encoder.pth \
      --label "SA-JEPA" --images a.jpg b.jpg c.jpg d.jpg e.jpg
  python visualize_attention.py --ckpt ... --image_dir /home/pai-ng/Jamal/XJTU-UP --n 5
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from models import ContextEncoder

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ───────────────────────── checkpoint ─────────────────────────
def _is_state_dict(d):
    return isinstance(d, dict) and len(d) > 0 and all(torch.is_tensor(v) for v in d.values())


def load_state_dict(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                   # torch without the weights_only argument
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    if isinstance(obj, dict) and not _is_state_dict(obj):
        for k in ("state_dict", "model_state_dict", "model", "context_encoder"):
            if k in obj:
                obj = obj[k].state_dict() if isinstance(obj[k], torch.nn.Module) else obj[k]
                break
    if not _is_state_dict(obj):
        keys = list(obj.keys())[:10] if isinstance(obj, dict) else type(obj)
        raise SystemExit(f"Could not find a state dict in {path}. Top-level content: {keys}")
    for prefix in ("module.", "context_encoder."):      # NOT 'encoder.': ContextEncoder's own keys start with it
        if all(k.startswith(prefix) for k in obj):
            obj = {k[len(prefix):]: v for k, v in obj.items()}
    return obj


def infer_arch(sd):
    missing = [k for k in ("proj.weight", "pos_embed", "norm.weight", "encoder.layers.0.linear1.weight") if k not in sd]
    if missing:
        raise SystemExit(f"This is not a ContextEncoder state dict (missing {missing}). First keys: {list(sd)[:8]}")
    D, c, ph, pw = sd["proj.weight"].shape
    P = sd["pos_embed"].shape[1]
    g = int(round(math.sqrt(P)))
    if c != 3 or ph != pw or g * g != P or sd["pos_embed"].shape[2] != D:
        raise SystemExit(f"Unexpected shapes: proj {tuple(sd['proj.weight'].shape)}, pos_embed {tuple(sd['pos_embed'].shape)}")
    layer_ids = sorted({int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")})
    if layer_ids != list(range(len(layer_ids))):
        raise SystemExit(f"Non-contiguous layer indices: {layer_ids}")
    return dict(embed_dim=D, patch=ph, grid=g, depth=len(layer_ids),
                mlp_ratio=sd["encoder.layers.0.linear1.weight"].shape[0] / D)


def build_model(sd, num_heads=None):
    a = infer_arch(sd)
    S = a["patch"] * a["grid"]
    model = ContextEncoder((S, S), a["grid"], a["embed_dim"], depth=a["depth"],
                           num_heads=num_heads, mlp_ratio=a["mlp_ratio"])
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise SystemExit(f"Checkpoint does not match the ContextEncoder built from it:\n{e}")
    return model.eval(), a, S


# ───────────────────────── attention + rollout ─────────────────────────
@torch.no_grad()
def forward_with_attn(model, x):
    """Layer-by-layer forward of ContextEncoder on the FULL patch set that also returns every layer's
    attention (B, heads, N, N). nn.TransformerEncoderLayer calls attention with need_weights=False, so the
    weights cannot be read with hooks; this reproduces the layer maths (verified against the real forward)."""
    z = model.proj(x).flatten(2).transpose(1, 2) + model.pos_embed
    attns = []
    for l in model.encoder.layers:
        if l.norm_first:
            xn = l.norm1(z)
            a, w = l.self_attn(xn, xn, xn, need_weights=True, average_attn_weights=False)
            z = z + l.dropout1(a)
            z = z + l._ff_block(l.norm2(z))
        else:
            a, w = l.self_attn(z, z, z, need_weights=True, average_attn_weights=False)
            z = l.norm1(z + l.dropout1(a))
            z = l.norm2(z + l._ff_block(z))
        attns.append(w)
    return model.norm(z), attns


def rollout(attns):
    """attns: list of (B, heads, N, N) -> R (B, N, N) with R = A^_L ... A^_1."""
    B, _, N, _ = attns[0].shape
    I = torch.eye(N, dtype=attns[0].dtype, device=attns[0].device)
    R = I.expand(B, N, N).clone()
    for w in attns:
        a = 0.5 * w.mean(1) + 0.5 * I
        a = a / a.sum(-1, keepdim=True)
        R = a @ R
    return R


def upsample_maps(maps, S):
    """maps: (B, g, g) -> (B, S, S): bilinear upsampling, then per-map min-max normalisation to [0, 1]."""
    up = F.interpolate(maps.unsqueeze(1), size=(S, S), mode="bilinear", align_corners=False).squeeze(1)
    lo, hi = up.amin((1, 2), keepdim=True), up.amax((1, 2), keepdim=True)
    return (up - lo) / (hi - lo + 1e-12)


def normalized_entropy(contrib):
    """contrib: (B, N) rows summing to 1 -> entropy / log(N); 1.0 = perfectly uniform."""
    return -(contrib * (contrib + 1e-12).log()).sum(-1) / math.log(contrib.shape[-1])


def pca_scores(tokens, k=3):
    """tokens: (B, N, D). Joint PCA over all B*N tokens (SVD of the centred matrix).
    Returns scores (B, N, k), explained-variance ratios (k,), and positional shares (k,).
    Component signs are fixed deterministically (largest-|loading| entry positive)."""
    B, N, D = tokens.shape
    if center_per_image:                      # drop the global (colour / illumination) component
        tokens = tokens - tokens.mean(1, keepdim=True)
    X = tokens.reshape(B * N, D).double()
    Xc = X - X.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(B * N - 1, 1)
    ratio = var[:k] / (var.sum() + 1e-12)
    comp = Vh[:k]
    sign = torch.sign(comp[torch.arange(k), comp.abs().argmax(1)])
    sign[sign == 0] = 1.0
    comp = comp * sign[:, None]
    P = (Xc @ comp.T).reshape(B, N, k)
    if B >= 2:      # variance of the per-position means / total variance (population variances => within [0, 1])
        pos_share = P.mean(0).var(0, unbiased=False) / (P.reshape(-1, k).var(0, unbiased=False) + 1e-12)
    else:
        pos_share = torch.full((k,), float("nan"), dtype=P.dtype)
    return P.float(), ratio.float(), pos_share.float()


def pca_rgb(tokens, grid, clip=2.0):
    """tokens: (B, N, D) -> rgb (B, grid, grid, 3) in [0, 1], variance ratios (3,), positional shares (3,).
    Each component is clipped to its [clip, 100-clip] percentile over ALL patches of ALL images, then scaled to [0, 1]."""
    B, N, _ = tokens.shape
    assert N == grid * grid, f"{N} tokens do not form a {grid}x{grid} grid"
    P, ratio, pos_share = pca_scores(tokens, 3)
    flat = P.reshape(-1, 3)
    lo = torch.quantile(flat, clip / 100.0, dim=0)
    hi = torch.quantile(flat, 1.0 - clip / 100.0, dim=0)
    rgb = ((flat - lo) / (hi - lo + 1e-12)).clamp(0, 1)
    return rgb.reshape(B, grid, grid, 3), ratio, pos_share


# ───────────────────────── images ─────────────────────────
def pick_images(args):
    if args.images:
        paths = list(args.images)
    else:
        found = sorted(p for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
                       if p.lower().endswith(IMG_EXTS))
        if len(found) < args.n:
            raise SystemExit(f"Only {len(found)} images under {args.image_dir}, need {args.n}.")
        idx = np.linspace(0, len(found) - 1, args.n).round().astype(int)    # evenly spread, deterministic
        paths = [found[i] for i in idx]
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Image not found: {p}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="state dict of the context encoder (.pth)")
    ap.add_argument("--label", default=None, help="name shown in the figure (default: checkpoint file name)")
    ap.add_argument("--images", nargs="*", default=None, help="explicit image paths (takes priority)")
    ap.add_argument("--image_dir", default=None, help="folder searched recursively when --images is not given")
    ap.add_argument("--n", type=int, default=5, help="how many images to take from --image_dir")
    ap.add_argument("--out", default="attention_map.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_heads", type=int, default=None,
                    help="default: ContextEncoder's own rule max(4, embed_dim//32). The head count cannot be read "
                         "from the weights, so override this only if the model was trained with another value.")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--cmap", default="jet")
    a = ap.parse_args()
    if not a.images and not a.image_dir:
        raise SystemExit("Give --images ... or --image_dir ...")

    label = a.label or os.path.splitext(os.path.basename(a.ckpt))[0]
    sd = load_state_dict(a.ckpt)
    model, arch, S = build_model(sd, a.num_heads)
    model = model.to(a.device)
    heads = model.encoder.layers[0].self_attn.num_heads
    g, N = arch["grid"], arch["grid"] ** 2
    print(f"checkpoint: {a.ckpt}\n  inferred: embed_dim={arch['embed_dim']} grid={g}x{g} patch={arch['patch']}px "
          f"image={S}x{S} depth={arch['depth']} mlp_ratio={arch['mlp_ratio']:g} | heads={heads} "
          f"({'--num_heads' if a.num_heads else 'ContextEncoder default rule'})")
    print(f"  label shown in the figure: '{label}'  (a checkpoint cannot tell JEPA from SA-JEPA: same architecture)")

    paths = pick_images(a)
    tf = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])      # same as the training/eval pipeline
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(a.device)
    B = x.shape[0]

    out, attns = forward_with_attn(model, x)
    with torch.no_grad():
        ref = model(x, [torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)])   # how FeatureExtractor calls it
    diff = (out - ref).abs().max().item()
    assert diff < 1e-4, f"manual forward differs from ContextEncoder.forward (max |diff| = {diff})"
    assert all(torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4) for w in attns), "attention rows must sum to 1"
    R = rollout(attns)
    contrib = R.mean(1)                                                                # (B, N): mean pooling
    assert torch.allclose(R.sum(-1), torch.ones_like(R.sum(-1)), atol=1e-4)
    assert torch.allclose(contrib.sum(-1), torch.ones(B, device=contrib.device), atol=1e-4)
    print(f"  checks passed: manual forward == ContextEncoder.forward (max |diff| = {diff:.1e}); attention/rollout rows sum to 1")

    maps = contrib.reshape(B, g, g).cpu()
    up = upsample_maps(maps, S)
    ent = normalized_entropy(contrib).cpu()                                            # 1.0 = perfectly uniform map
    disp = (x.cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    print(f"\n  {'image':<34}{'H/Hmax':>8}   peak patch (row,col)   peak share")
    for i, p in enumerate(paths):
        k = int(contrib[i].argmax())
        print(f"  {os.path.basename(p)[:33]:<34}{ent[i]:>8.3f}   ({k // g},{k % g}){'':<14}{contrib[i, k]:.3f}")
    print("  H/Hmax near 1.0 means the map is almost uniform (little structure to interpret).")

    pca, ratio, pos_share = pca_rgb(out.cpu(), g)                                      # joint PCA over all images
    pca_up = F.interpolate(pca.permute(0, 3, 1, 2), size=(S, S), mode="nearest").permute(0, 2, 3, 1)   # 1 patch = 1 block
    pct = " / ".join(f"{100 * r:.1f}%" for r in ratio.tolist())
    pos = " / ".join(f"{v:.2f}" for v in pos_share.tolist()) if B >= 2 else "n/a (needs >= 2 images)"
    print(f"\n  PCA (joint over {B} images, {B * N} patches): variance explained PC1/PC2/PC3 = {pct}")
    print(f"  positional share PC1/PC2/PC3 = {pos}" + (f"   (chance ~ {1.0 / B:.2f}; near 1.0 = the component only encodes location)" if B >= 2 else ""))

    fig, ax = plt.subplots(3, B, figsize=(2.4 * B + 0.8, 8.0), squeeze=False)
    for i, p in enumerate(paths):
        ax[0, i].imshow(disp[i]); ax[0, i].set_title(os.path.basename(p)[:22], fontsize=7); ax[0, i].axis("off")
        ax[1, i].imshow(disp[i]); ax[1, i].imshow(up[i].numpy(), cmap=a.cmap, alpha=a.alpha, vmin=0, vmax=1)
        ax[1, i].set_title(f"H/Hmax = {ent[i]:.3f}", fontsize=7); ax[1, i].axis("off")
        ax[2, i].imshow(pca_up[i].numpy()); ax[2, i].axis("off")
    for r, t in enumerate(["Image", f"{label}\nattention", f"{label}\nPCA"]):
        ax[r, 0].text(-0.04, 0.5, t, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
    fig.suptitle("Row 2: patch contribution to the evaluation embedding (attention rollout), each map min-max normalised\n"
                 f"Row 3: PCA of final patch features, fitted jointly on all images; variance explained {pct}; "
                 f"positional share {pos}", fontsize=8)
    plt.tight_layout(rect=[0.02, 0, 1, 0.95], h_pad=2.0)
    plt.savefig(a.out, dpi=150, bbox_inches="tight", pad_inches=0.15)
    npz = os.path.splitext(a.out)[0] + ".npz"
    np.savez(npz, maps=maps.numpy(), paths=np.array(paths), entropy=ent.numpy(), label=label, grid=g,
             pca_rgb=pca.numpy(), pca_var_ratio=ratio.numpy(), pca_pos_share=pos_share.numpy())
    print(f"\nsaved {a.out} and {npz}")


if __name__ == "__main__":
    main()
