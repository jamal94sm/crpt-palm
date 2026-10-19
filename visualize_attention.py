"""visualize_attention.py -- LAST-LAYER attention maps of several trained JEPA-family encoders
(rows = methods) on k palm images (columns).

Default methods / checkpoint folder (the files written by --save_ckpt, one per method):
    <ckpt_dir>/ckpt_jepa_train_<label>.pth      -> row "JEPA"
    <ckpt_dir>/ckpt_dmtjepa_train_<label>.pth   -> row "DMT-JEPA"
    <ckpt_dir>/ckpt_palmjepa_train_<label>.pth  -> row "Palm-JEPA"
(the *_resume.pth files are ignored). They hold the evaluation (target) encoder.

WHAT IS SHOWN (default --map last)
The encoder has no CLS token and the evaluation embedding is the MEAN of the patch tokens
(FeatureExtractor). The map is the LAST encoder layer only, head-averaged, mean over the queries
("attention received"):
    A_bar[i,j] = mean over the heads of the last layer's attention (query i, key j)     (N x N)
    m_j        = (1/N) * sum_i A_bar[i,j]                                                (sums to 1)
reshaped row-major to (grid, grid), bilinearly upsampled to the image and min-max normalised PER MAP.
This is the closest analogue of "the averaged maps from the heads of the last attention layer" of the
DMT-JEPA paper for an encoder WITHOUT a CLS token; the paper does not say how the N x N matrix is reduced
to one map, so the mean over the queries is an inference, not a documented step.

--map rollout shows instead attention rollout over ALL layers (Abnar & Zuidema, 2020):
    A^_l = row-normalise(0.5 * mean_heads(A_l) + 0.5 * I),  R = A^_L ... A^_1,  c_j = (1/N) sum_i R[i,j].

Run from the repo root (models.py importable):
  python visualize_attention.py                                  # defaults below
  python visualize_attention.py --ckpt_dir out_xd_xpalm_xjtu_to_casia_aug8 --k 5 \
      --image_dir /home/pai-ng/Jamal/CASIA-MS-ROI --out attention_methods.png
  python visualize_attention.py --images a.jpg b.jpg c.jpg --methods jepa dmtjepa palmjepa
  python visualize_attention.py --methods "JEPA=/path/a.pth" "Palm-JEPA=/path/b.pth"   # explicit files
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
DEFAULT_CKPT_DIR = "out_xd_xpalm_xjtu_to_casia_aug8"
DEFAULT_IMAGE_DIR = "/home/pai-ng/Jamal/CASIA-MS-ROI"
DEFAULT_METHODS = ["jepa", "dmtjepa", "palmjepa"]
DISPLAY_NAME = {"jepa": "JEPA", "dmtjepa": "DMT-JEPA", "palmjepa": "Palm-JEPA"}


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
        raise SystemExit(f"This is not a ContextEncoder/TargetEncoder state dict (missing {missing}). First keys: {list(sd)[:8]}")
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


def last_layer_map(attns):
    """Last encoder layer only: average the heads, then average over the QUERY tokens (attention received).
    attns: list of (B, heads, N_query, N_key) -> (B, N_key); sums to 1 because every attention row does."""
    return attns[-1].mean(1).mean(1)


def upsample_maps(maps, S):
    """maps: (B, g, g) -> (B, S, S): bilinear upsampling, then per-map min-max normalisation to [0, 1]."""
    up = F.interpolate(maps.unsqueeze(1), size=(S, S), mode="bilinear", align_corners=False).squeeze(1)
    lo, hi = up.amin((1, 2), keepdim=True), up.amax((1, 2), keepdim=True)
    return (up - lo) / (hi - lo + 1e-12)


def normalized_entropy(contrib):
    """contrib: (B, N) rows summing to 1 -> entropy / log(N); 1.0 = perfectly uniform."""
    return -(contrib * (contrib + 1e-12).log()).sum(-1) / math.log(contrib.shape[-1])


# ───────────────────────── methods / checkpoints ─────────────────────────
def find_ckpt(ckpt_dir, tag, train_label=None):
    """ckpt_<tag>_train_<label>.pth inside ckpt_dir (the *_resume.pth files are ignored)."""
    hits = sorted(p for p in glob.glob(os.path.join(ckpt_dir, f"ckpt_{tag}_train_*.pth"))
                  if not p.endswith("_resume.pth"))
    if train_label:
        hits = [p for p in hits if os.path.basename(p) == f"ckpt_{tag}_train_{train_label}.pth"]
    return hits


def resolve_methods(args):
    """-> list of (label, path). Items of --methods are
         tag                (jepa | dmtjepa | palmjepa | any tag in the folder)  -> found in --ckpt_dir
         tag=Label          same, with a custom row name
         Label=/path/x.pth  an explicit checkpoint file
         /path/x.pth        an explicit checkpoint file (row name = file name)"""
    specs, missing = [], []
    for item in args.methods:
        left, right = (item.split("=", 1) + [None])[:2] if "=" in item else (item, None)
        if right is not None and right.lower().endswith(".pth"):
            label, path = left, right
        elif left.lower().endswith(".pth"):
            label, path = os.path.splitext(os.path.basename(left))[0], left
        else:
            tag = left
            label = right or DISPLAY_NAME.get(tag.lower().replace("-", "").replace("_", ""), tag)
            hits = find_ckpt(args.ckpt_dir, tag.lower().replace("-", "").replace("_", ""), args.train_label)
            if not hits:
                missing.append(f"{tag}: no ckpt_{tag}_train_*.pth in {args.ckpt_dir}")
                continue
            if len(hits) > 1:
                print(f"  [{tag}] {len(hits)} checkpoints match; using {os.path.basename(hits[0])} "
                      f"(choose another with --train_label). Others: {[os.path.basename(h) for h in hits[1:]]}")
            path = hits[0]
        if not os.path.isfile(path):
            missing.append(f"{label}: file not found {path}")
            continue
        specs.append((label, path))
    for m in missing:
        print(f"  WARNING - skipped, {m}")
    if not specs:
        have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.ckpt_dir, "*.pth")))
        raise SystemExit(f"No checkpoint found. Files in {args.ckpt_dir}: {have}")
    return specs


# ───────────────────────── images ─────────────────────────
def pick_images(args):
    if args.images:
        paths = list(args.images)
    else:
        found = sorted(p for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
                       if p.lower().endswith(IMG_EXTS))
        if len(found) < args.k:
            raise SystemExit(f"Only {len(found)} images under {args.image_dir}, need {args.k}.")
        if args.pick == "random":
            idx = sorted(np.random.RandomState(args.seed).choice(len(found), args.k, replace=False).tolist())
        else:                                                           # evenly spread, deterministic
            idx = np.linspace(0, len(found) - 1, args.k).round().astype(int)
        paths = [found[i] for i in idx]
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Image not found: {p}")
    return paths


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Last-layer attention maps: methods (rows) x palm images (columns)")
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="folder with the ckpt_<method>_train_<label>.pth files")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help="rows, top to bottom: tags (jepa dmtjepa palmjepa), tag=Label, Label=file.pth or file.pth")
    ap.add_argument("--train_label", default=None,
                    help="only needed when a method has several checkpoints, e.g. 'xpalm+xjtu'")
    ap.add_argument("--k", "--n", dest="k", type=int, default=5, help="number of palm images (columns)")
    ap.add_argument("--images", nargs="*", default=None, help="explicit image paths (takes priority over --image_dir)")
    ap.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="folder searched recursively when --images is not given")
    ap.add_argument("--pick", choices=["spread", "random"], default="spread",
                    help="how k images are taken from --image_dir: evenly spread over the sorted list, or random (--seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", choices=["last", "rollout"], default="last",
                    help="last = last layer only (default); rollout = attention rollout over all layers")
    ap.add_argument("--out", default="attention_methods.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_heads", type=int, default=None,
                    help="default: ContextEncoder's own rule max(4, embed_dim//32). The head count cannot be read "
                         "from the weights, so override this only if the models were trained with another value.")
    ap.add_argument("--alpha", type=float, default=0.55, help="opacity of the attention overlay")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--no_image_row", action="store_true", help="do not draw the row with the plain images")
    ap.add_argument("--show_entropy", action="store_true", help="print H/Hmax (1 = uniform map) above every map")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    if a.k < 1:
        raise SystemExit("--k must be >= 1")
    if not a.images and not a.image_dir:
        raise SystemExit("Give --images ... or --image_dir ...")

    specs = resolve_methods(a)
    paths = pick_images(a)
    pil = [Image.open(p).convert("RGB") for p in paths]
    B = len(paths)
    print(f"\nimages ({B}):")
    for p in paths:
        print(f"  {p}")

    rows = []                                           # (label, maps (B,g,g), up (B,D,D), entropy (B,))
    archs = {}
    for label, path in specs:
        sd = load_state_dict(path)
        model, arch, S = build_model(sd, a.num_heads)
        model = model.to(a.device)
        heads = model.encoder.layers[0].self_attn.num_heads
        g, N = arch["grid"], arch["grid"] ** 2
        archs[label] = (S, g)
        print(f"\n[{label}] {path}\n  inferred: embed_dim={arch['embed_dim']} grid={g}x{g} patch={arch['patch']}px "
              f"image={S}x{S} depth={arch['depth']} mlp_ratio={arch['mlp_ratio']:g} | heads={heads} "
              f"({'--num_heads' if a.num_heads else 'ContextEncoder default rule'})")
        tf = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)])   # same as the training/eval pipeline
        x = torch.stack([tf(im) for im in pil]).to(a.device)

        out, attns = forward_with_attn(model, x)
        with torch.no_grad():
            ref = model(x, [torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)])   # how FeatureExtractor calls it
        diff = (out - ref).abs().max().item()
        assert diff < 1e-4, f"[{label}] manual forward differs from ContextEncoder.forward (max |diff| = {diff})"
        assert all(torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4) for w in attns), \
            "attention rows must sum to 1"
        if a.map == "rollout":
            R = rollout(attns)
            m = R.mean(1)                                                       # (B, N): mean pooling
        else:
            m = last_layer_map(attns)                                           # (B, N): last layer, heads+queries averaged
        assert (m >= 0).all() and torch.allclose(m.sum(-1), torch.ones(B, device=m.device), atol=1e-4)
        print(f"  checks passed: manual forward == ContextEncoder.forward (max |diff| = {diff:.1e}); "
              f"attention rows sum to 1; map sums to 1")
        ent = normalized_entropy(m).cpu()
        for i, p in enumerate(paths):
            k1 = int(m[i].argmax())
            print(f"    {os.path.basename(p)[:34]:<36} H/Hmax = {ent[i]:.3f}   peak patch (row,col) = ({k1 // g},{k1 % g})")
        rows.append((label, m.reshape(B, g, g).cpu(), ent))

    D = max(S for S, _ in archs.values())               # common display size
    disp = np.stack([np.asarray(im.resize((D, D), Image.BICUBIC), dtype=np.float32) / 255.0 for im in pil])
    ups = [(lab, upsample_maps(mp, D).numpy(), ent, mp) for lab, mp, ent in rows]

    R_ = len(ups) + (0 if a.no_image_row else 1)
    fig, ax = plt.subplots(R_, B, figsize=(2.15 * B + 1.3, 2.15 * R_ + 0.7), squeeze=False)
    r0 = 0
    if not a.no_image_row:
        for i, p in enumerate(paths):
            ax[0, i].imshow(disp[i]); ax[0, i].set_title(os.path.basename(p)[:22], fontsize=7); ax[0, i].axis("off")
        ax[0, 0].text(-0.05, 0.5, "Image", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right",
                      fontsize=11, fontweight="bold")
        r0 = 1
    for r, (lab, up, ent, _) in enumerate(ups, start=r0):
        for i in range(B):
            ax[r, i].imshow(disp[i])
            ax[r, i].imshow(up[i], cmap=a.cmap, alpha=a.alpha, vmin=0, vmax=1)
            if a.show_entropy:
                ax[r, i].set_title(f"H/Hmax = {ent[i]:.3f}", fontsize=6)
            ax[r, i].axis("off")
        ax[r, 0].text(-0.05, 0.5, lab, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right",
                      fontsize=11, fontweight="bold")
    what = ("last encoder layer, averaged over heads and queries (attention received)" if a.map == "last"
            else "attention rollout over all layers")
    fig.suptitle(f"Attention maps - {what}   [each map min-max normalised]", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.92 if a.show_entropy else 0.94, bottom=0.02,
                        wspace=0.03, hspace=0.05 if not a.show_entropy else 0.18)
    cax = fig.add_axes([0.92, 0.2, 0.014, 0.6])
    sm = plt.cm.ScalarMappable(cmap=a.cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("relative attention", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    plt.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.15)
    npz = os.path.splitext(a.out)[0] + ".npz"
    np.savez(npz, paths=np.array(paths), methods=np.array([r[0] for r in rows]), map_type=a.map,
             **{f"maps__{lab}": mp.numpy() for lab, mp, _ in rows},
             **{f"entropy__{lab}": ent.numpy() for lab, _, ent in rows})
    print(f"\nsaved {a.out} and {npz}")


if __name__ == "__main__":
    main()
