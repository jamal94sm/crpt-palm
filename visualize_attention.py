"""visualize_attention.py -- LAST-LAYER attention maps of several trained JEPA-family encoders
(rows = methods) on k palm images (columns).

Default methods (rows, top to bottom) and checkpoint folder (the files written by --save_ckpt, one per method):
    <ckpt_dir>/ckpt_vicreg_train_<label>.pth    -> row "VICReg"
    <ckpt_dir>/ckpt_maskfeat_train_<label>.pth  -> row "MaskFeat"
    <ckpt_dir>/ckpt_dinov2_train_<label>.pth    -> row "DINOv2"
    <ckpt_dir>/ckpt_jepa_train_<label>.pth      -> row "JEPA"
    <ckpt_dir>/ckpt_cjepa_train_<label>.pth     -> row "C-JEPA"
    <ckpt_dir>/ckpt_dmtjepa_train_<label>.pth   -> row "DMT-JEPA"
    <ckpt_dir>/ckpt_palmjepa_train_<label>.pth  -> row "Palm-JEPA"
(the *_resume.pth files are ignored; *_best.pth / *_last.pth only with --ckpt_which). They hold the evaluation
encoder. VICReg and MaskFeat use the same ViT as the JEPA encoders (no CLS token, mean-pooled patch tokens), so
their rows show the same quantity; MaskFeat's unused [MASK] token is dropped and its forward is checked against
the real MaskedViT.

CONDITIONS (one output figure per condition)
--conditions clean poor_light wet device   (default)  ->  <out>_clean.png, <out>_poor_light.png, <out>_wet.png,
<out>_device.png (+ one .npz each). Every model sees the SAME corrupted images. Conditions:
    clean | poor_light | strong_light | wet | dry | dirt | device (= device change: colour cast, resolution,
    sharpness, local contrast, noise, JPEG) | any operator name of corruption_types.py
    (side_light glare cast_shadow colored_light haze light_knob dev_color_cast dev_resolution dev_detail
     dev_local_contrast dev_sensor_noise dev_jpeg dev_device_profile)
The corruptions are the ones of the corruption-type ablation (corruption_types.py), applied to the image before the
encoder, with a fixed seed (--cond_seed) and strength (--severity, --light_knob). --list_conditions prints the names.
Rows: --methods / --skip choose the baselines (VICReg and MaskFeat are in the default list).

WHICH ROWS: --methods selects and orders them (a subset is fine), --skip removes some from the default list,
--ckpt_dirs TAG=DIR takes a method's checkpoint from another folder. FONT SIZE: --font_scale (all text),
--label_size / --title_size / --cbar_size (individually), --cell (inches per image).

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
  python visualize_attention.py --skip vicreg maskfeat                  # all default rows except these
  python visualize_attention.py --methods dinov2 jepa palmjepa --font_scale 1.8
  python visualize_attention.py --ckpt_dir A --ckpt_dirs palmjepa=B dinov2=C --ckpt_which last
  python visualize_attention.py --ckpt_dir A --conditions clean poor_light wet device --methods vicreg maskfeat jepa palmjepa
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
DEFAULT_METHODS = ["vicreg", "maskfeat", "dinov2", "jepa", "cjepa", "dmtjepa", "palmjepa"]
DISPLAY_NAME = {"vicreg": "VICReg", "maskfeat": "MaskFeat", "dinov2": "DINOv2", "jepa": "JEPA",
                "cjepa": "C-JEPA", "dmtjepa": "DMT-JEPA", "palmjepa": "Palm-JEPA"}


def _norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")


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


# ───────────────────────── DINOv2 backbone (CLS token, blocks.*) ─────────────────────────
def _palm_module(subdir, name):
    import importlib.util          # load <subdir>/models.py under another name: it clashes with models.py
    for base in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        p = os.path.join(base, subdir, "models.py")
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(name, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(f"{subdir}/models.py not found (run from the repo root)")


def _dinov2_module():
    return _palm_module("dinov2-palm", "dinov2_palm_models")


def build_dino(sd, num_heads=None):
    D, c, ph, pw = sd["patch_embed.weight"].shape
    P = sd["pos_embed"].shape[1] - 1                       # minus the CLS position
    g = int(round(math.sqrt(P)))
    if c != 3 or ph != pw or g * g != P:
        raise SystemExit(f"Unexpected DINOv2 shapes: patch_embed {tuple(sd['patch_embed.weight'].shape)}, "
                         f"pos_embed {tuple(sd['pos_embed'].shape)}")
    depth = len({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    arch = dict(embed_dim=D, patch=ph, grid=g, depth=depth,
                mlp_ratio=sd["blocks.0.mlp.fc1.weight"].shape[0] / D)
    ls = 1e-5 if "blocks.0.ls1.gamma" in sd else 0.0
    S = ph * g
    model = _dinov2_module().DinoV2ViT(S, g, D, depth=depth, num_heads=num_heads,
                                       mlp_ratio=arch["mlp_ratio"], layerscale_init=ls)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise SystemExit(f"Checkpoint does not match the DinoV2ViT built from it:\n{e}")
    return model.eval(), arch, S


@torch.no_grad()
def forward_with_attn_dino(model, x):
    """DinoV2ViT forward that also returns every block's attention (B, heads, N+1, N+1), CLS = index 0."""
    z = model.patch_embed(x).flatten(2).transpose(1, 2)
    z = torch.cat([model.cls_token.expand(z.shape[0], -1, -1), z], dim=1)
    z = z + model._pos(z.shape[1] - 1)
    attns = []
    for blk in model.blocks:
        at = blk.attn
        B, N, C = z.shape
        hd = C // at.num_heads
        qkv = at.qkv(blk.norm1(z)).reshape(B, N, 3, at.num_heads, hd).permute(2, 0, 3, 1, 4)
        w = ((qkv[0] @ qkv[1].transpose(-2, -1)) * hd ** -0.5).softmax(-1)
        z = z + blk.drop_path(blk.ls1(at.proj((w @ qkv[2]).transpose(1, 2).reshape(B, N, C))))
        z = z + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(z))))
        attns.append(w)
    return model.norm(z), attns


def dino_map(attns, kind, query):
    """(B, N_patches) map summing to 1. query='cls': what the CLS token (= the evaluation embedding)
    attends to; query='patches': mean over the patch queries, i.e. 'attention received', the same
    quantity as the other rows. kind='rollout' uses attention rollout over all blocks instead of the last one."""
    if kind == "rollout":
        R = rollout(attns)
        m = R[:, 0, 1:] if query == "cls" else R[:, 1:, 1:].mean(1)
    else:
        A = attns[-1].mean(1)
        m = A[:, 0, 1:] if query == "cls" else A[:, 1:, 1:].mean(1)
    return m / m.sum(-1, keepdim=True)


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
def find_ckpt(ckpt_dir, tag, train_label=None, which="plain"):
    """ckpt_<tag>_train_<label>[_best|_last].pth inside ckpt_dir. which='plain' (default) = the file without a
    suffix; 'best' / 'last' = the *_best.pth / *_last.pth files. The *_resume.pth files are always ignored."""
    sfx = f"_{which}" if which in ("best", "last") else ""
    hits = sorted(p for p in glob.glob(os.path.join(ckpt_dir, f"ckpt_{tag}_train_*{sfx}.pth"))
                  if not p.endswith("_resume.pth")
                  and (sfx or not p.endswith(("_best.pth", "_last.pth"))))
    if train_label:
        hits = [p for p in hits if os.path.basename(p) == f"ckpt_{tag}_train_{train_label}{sfx}.pth"]
    return hits


def resolve_methods(args):
    """-> list of (label, path). Items of --methods are
         tag                (jepa | dmtjepa | palmjepa | any tag in the folder)  -> found in --ckpt_dir
         tag=Label          same, with a custom row name
         Label=/path/x.pth  an explicit checkpoint file
         /path/x.pth        an explicit checkpoint file (row name = file name)
    --skip drops rows (by tag or label); --ckpt_dirs TAG=DIR overrides --ckpt_dir for one method."""
    specs, missing = [], []
    skip = {_norm(x) for x in (args.skip or [])}
    dirs = {}
    for kv in (args.ckpt_dirs or []):
        if "=" not in kv:
            raise SystemExit(f"--ckpt_dirs expects TAG=DIR, got '{kv}'")
        k, v = kv.split("=", 1)
        dirs[_norm(k)] = v
    for item in args.methods:
        left, right = (item.split("=", 1) + [None])[:2] if "=" in item else (item, None)
        if _norm(left) in skip or (right is not None and _norm(right) in skip):
            print(f"  [{left}] skipped (--skip)")
            continue
        if right is not None and right.lower().endswith(".pth"):
            label, path = left, right
        elif left.lower().endswith(".pth"):
            label, path = os.path.splitext(os.path.basename(left))[0], left
        else:
            tag = left
            label = right or DISPLAY_NAME.get(_norm(tag), tag)
            cdir = dirs.get(_norm(tag), args.ckpt_dir)
            hits = find_ckpt(cdir, _norm(tag), args.train_label, args.ckpt_which)
            if not hits:
                missing.append(f"{tag}: no ckpt_{_norm(tag)}_train_*"
                               f"{'_' + args.ckpt_which if args.ckpt_which != 'plain' else ''}.pth in {cdir}")
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


# ───────────────────────── conditions (clean / corrupted inputs) ─────────────────────────
COND_DISPLAY = {"clean": "Clean", "poor_light": "Poor lighting", "strong_light": "Strong lighting",
                "wet": "Wet", "dry": "Dry", "dirt": "Dirt", "device": "Device change"}
COND_ALIAS = {"clean": None, "poor_light": ("light_knob", -1), "strong_light": ("light_knob", +1),
              "wet": ("wet", None), "dry": ("dry", None), "dirt": ("dirt", None),
              "device": ("dev_device_profile", None)}


def _ops():
    import corruption_types as T
    return T


def condition_names():
    T = _ops()
    return list(dict.fromkeys(list(COND_ALIAS) + T.LIGHTING + T.SURFACE + T.DEVICE))


def apply_condition(x01, cond, a):
    """x01: (B,3,S,S) in [0,1] -> corrupted images in [0,1]. Deterministic: seeded with --cond_seed, so every model
    (and the displayed background) sees identical corruption parameters."""
    key = _norm(cond)
    names = {_norm(n): n for n in condition_names()}
    if key not in names:
        raise SystemExit(f"Unknown condition '{cond}'. Available: {condition_names()}")
    name = names[key]
    if name == "clean":
        return x01
    T = _ops()
    torch.manual_seed(a.cond_seed)
    op, sign = COND_ALIAS.get(name, (name, None))
    fn = getattr(T, op)
    if op == "light_knob":
        k = None if sign is None else sign * a.light_knob          # fixed knob: poor (-) / strong (+) lighting
        out = fn(x01, a.severity, a.light_knob, k)
    else:
        out = fn(x01, a.severity)
    return out.clamp(0, 1)


# ───────────────────────── models ─────────────────────────
def load_models(specs, a):
    """Build every method's encoder once. -> list of dict(label, path, model, is_dino, is_maskfeat, arch, S, g, heads)."""
    ms = []
    for label, path in specs:
        sd = load_state_dict(path)
        is_dino = "cls_token" in sd                      # DINOv2 backbone
        is_maskfeat = ("mask_token" in sd) and not is_dino                 # MaskFeat's MaskedViT: ContextEncoder + an unused [MASK] token
        if is_maskfeat:
            sd = {k: v for k, v in sd.items() if k != "mask_token"}
        model, arch, S = (build_dino if is_dino else build_model)(sd, a.num_heads)
        model = model.to(a.device)
        heads = model.blocks[0].attn.num_heads if is_dino else model.encoder.layers[0].self_attn.num_heads
        print(f"\n[{label}] {path}\n  inferred: embed_dim={arch['embed_dim']} grid={arch['grid']}x{arch['grid']} "
              f"patch={arch['patch']}px image={S}x{S} depth={arch['depth']} mlp_ratio={arch['mlp_ratio']:g} | "
              f"heads={heads} ({'--num_heads' if a.num_heads else 'ContextEncoder default rule'})")
        ms.append(dict(label=label, path=path, model=model, is_dino=is_dino, is_maskfeat=is_maskfeat,
                       arch=arch, S=S, g=arch["grid"], heads=heads))
    return ms


def compute_rows(ms, pil, paths, cond, a, check):
    """-> rows [(label, maps (B,g,g), entropy (B,))] for one condition. check=True also verifies the manual
    forward against the real one (done for the first condition only)."""
    rows = []
    B = len(paths)
    for m_ in ms:
        label, model, arch, S, g = m_["label"], m_["model"], m_["arch"], m_["S"], m_["g"]
        is_dino, is_maskfeat, N = m_["is_dino"], m_["is_maskfeat"], m_["g"] ** 2
        x01 = torch.stack([transforms.ToTensor()(im.resize((S, S), Image.BILINEAR)) for im in pil]).to(a.device)
        x01 = apply_condition(x01, cond, a)
        x = (x01 - 0.5) / 0.5                                   # same normalisation as the training/eval pipeline
        with torch.no_grad():
            if is_dino:
                out, attns = forward_with_attn_dino(model, x)
                diff = 0.0
                if check:
                    import copy
                    ref = model.forward_features(x)
                    diff32 = max((out[:, 0] - ref["cls"]).abs().max().item(),
                                 (out[:, 1:] - ref["patch"]).abs().max().item())
                    # The real model uses the fused scaled_dot_product_attention, the manual one matmul+softmax: in
                    # float32 they differ by rounding noise that grows with the activation scale of trained weights.
                    # So the equivalence is checked in float64, where an actual maths mismatch cannot hide.
                    m64, x64 = copy.deepcopy(model).double(), x.double()
                    o64, _ = forward_with_attn_dino(m64, x64)
                    r64 = m64.forward_features(x64)
                    diff = max((o64[:, 0] - r64["cls"]).abs().max().item(),
                               (o64[:, 1:] - r64["patch"]).abs().max().item())
                    print(f"  [{label}] DINOv2 float32 manual-vs-fused max |diff| = {diff32:.1e} (rounding); float64 = {diff:.1e}")
                    del m64, x64
            else:
                out, attns = forward_with_attn(model, x)
                diff = 0.0
                if check:
                    ref = model(x, [torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)])   # how FeatureExtractor calls it
                    diff = (out - ref).abs().max().item()
                    if is_maskfeat:                          # also against MaskFeat's own forward (mask=None, as at evaluation)
                        sd_full = load_state_dict(m_["path"])
                        nat = _palm_module("maskfeat-palm", "maskfeat_palm_models").MaskedViT(
                            (S, S), g, arch["embed_dim"], depth=arch["depth"], num_heads=m_["heads"], mlp_ratio=arch["mlp_ratio"])
                        nat.load_state_dict(sd_full, strict=True)
                        nat = nat.to(a.device).eval()
                        d_nat = (out - nat(x, mask=None)).abs().max().item()
                        assert d_nat < 1e-4, f"[{label}] differs from MaskFeat's own forward (max |diff| = {d_nat})"
                        print(f"  [{label}] MaskFeat: [MASK] token dropped (unused at evaluation); max |diff| vs MaskedViT.forward = {d_nat:.1e}")
        if check:
            assert diff < (1e-8 if is_dino else 1e-4), f"[{label}] manual forward differs from the real forward (max |diff| = {diff})"
        assert all(torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4) for w in attns), \
            "attention rows must sum to 1"
        if is_dino:
            m = dino_map(attns, a.map, a.dino_query)
        elif a.map == "rollout":
            m = rollout(attns).mean(1)                                          # (B, N): mean pooling
        else:
            m = last_layer_map(attns)                                           # (B, N): last layer, heads+queries averaged
        assert (m >= 0).all() and torch.allclose(m.sum(-1), torch.ones(B, device=m.device), atol=1e-4)
        ent = normalized_entropy(m).cpu()
        print(f"  [{cond}] {label:<10} mean H/Hmax = {ent.mean():.3f}   peak patch (row,col) of image 1 = "
              f"({int(m[0].argmax()) // g},{int(m[0].argmax()) % g})")
        rows.append((label, m.reshape(B, g, g).cpu(), ent))
    return rows


def render_figure(rows, pil, paths, ms, cond, a, out):
    B = len(paths)
    archs = {m_["label"]: (m_["S"], m_["g"]) for m_ in ms}
    D = max(S for S, _ in archs.values())               # common display size
    base = torch.stack([transforms.ToTensor()(im.resize((D, D), Image.BICUBIC)) for im in pil])
    base = apply_condition(base, cond, a)               # the displayed background shows the corrupted input
    disp = base.permute(0, 2, 3, 1).numpy()
    ups = [(lab, upsample_maps(mp, D).numpy(), ent, mp) for lab, mp, ent in rows]

    R_ = len(ups) + (0 if a.no_image_row else 1)
    fs = a.font_scale
    lab_sz = a.label_size if a.label_size else 11 * fs
    ttl_sz = a.title_size if a.title_size else 6 * fs
    cb_sz = a.cbar_size if a.cbar_size else 8 * fs
    cell = a.cell
    # margins in inches, so that big fonts get the room they need
    left_in = 0.15 + lab_sz / 72 * 1.4
    right_in = 0.55 + cb_sz / 72 * 2.4
    gap_w = 0.04
    gap_h = 0.05 + (ttl_sz / 72 * 1.7 if a.show_entropy else 0.0)
    top_in, bot_in = 0.08, 0.08
    W = left_in + cell * B + gap_w * (B - 1) + right_in
    H = top_in + cell * R_ + gap_h * (R_ - 1) + bot_in
    longest = max(len(r[0]) for r in ups) if ups else 0
    if longest * lab_sz * 0.62 / 72 > cell * 1.1:
        print(f"  WARNING: the longest row name ({longest} chars at {lab_sz:.0f} pt) is taller than a cell "
              f"({cell:g} in); lower --label_size / --font_scale or raise --cell.")
    fig, ax = plt.subplots(R_, B, figsize=(W, H), squeeze=False)
    r0 = 0
    if not a.no_image_row:
        for i, p in enumerate(paths):
            ax[0, i].imshow(disp[i]); ax[0, i].axis("off")
        ax[0, 0].text(-0.03, 0.5, "Image", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right",
                      fontsize=lab_sz, fontweight="bold")
        r0 = 1
    for r, (lab, up, ent, _) in enumerate(ups, start=r0):
        for i in range(B):
            ax[r, i].imshow(disp[i])
            ax[r, i].imshow(up[i], cmap=a.cmap, alpha=a.alpha, vmin=0, vmax=1)
            if a.show_entropy:
                ax[r, i].set_title(f"H/Hmax = {ent[i]:.3f}", fontsize=ttl_sz, pad=2)
            ax[r, i].axis("off")
        ax[r, 0].text(-0.03, 0.5, lab, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right",
                      fontsize=lab_sz, fontweight="bold")
    if a.cond_title:
        fig.suptitle(COND_DISPLAY.get(_norm(cond) if _norm(cond) in COND_DISPLAY else cond, cond),
                     fontsize=lab_sz * 1.1, fontweight="bold", y=1.0)

    fig.subplots_adjust(left=left_in / W, right=1 - right_in / W, top=1 - top_in / H, bottom=bot_in / H,
                        wspace=gap_w / cell, hspace=gap_h / cell)
    cax = fig.add_axes([(W - right_in + 0.12) / W, 0.2, 0.16 / W, 0.6])
    sm = plt.cm.ScalarMappable(cmap=a.cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("relative attention", fontsize=cb_sz)
    cb.ax.tick_params(labelsize=cb_sz * 0.88)
    plt.savefig(out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    npz = os.path.splitext(out)[0] + ".npz"
    np.savez(npz, paths=np.array(paths), methods=np.array([r[0] for r in rows]), map_type=a.map, condition=cond,
             **{f"maps__{lab}": mp.numpy() for lab, mp, _ in rows},
             **{f"entropy__{lab}": ent.numpy() for lab, _, ent in rows})
    print(f"  saved {out} and {npz}")


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Last-layer attention maps: methods (rows) x palm images (columns), "
                                             "one figure per condition (clean / corrupted)")
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="folder with the ckpt_<method>_train_<label>.pth files")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help="rows, top to bottom: tags (vicreg maskfeat dinov2 jepa cjepa dmtjepa palmjepa), tag=Label, "
                         "Label=file.pth or file.pth")
    ap.add_argument("--skip", nargs="*", default=None, metavar="TAG",
                    help="rows to leave out of --methods (tags or labels, e.g. --skip vicreg maskfeat)")
    ap.add_argument("--ckpt_dirs", nargs="*", default=None, metavar="TAG=DIR",
                    help="checkpoint folder of single methods, overriding --ckpt_dir (e.g. palmjepa=/path/a dinov2=/path/b)")
    ap.add_argument("--ckpt_which", choices=["plain", "best", "last"], default="plain",
                    help="plain = ckpt_<tag>_train_<label>.pth (default); best / last = the *_best.pth / *_last.pth files")
    ap.add_argument("--train_label", default=None,
                    help="only needed when a method has several checkpoints, e.g. 'xpalm+xjtu'")
    ap.add_argument("--conditions", nargs="+", default=["clean", "poor_light", "wet", "device"], metavar="COND",
                    help="one output figure per condition: clean, poor_light, strong_light, wet, dry, dirt, device "
                         "(= device change) or any operator name (see --list_conditions). Default: clean poor_light wet device")
    ap.add_argument("--list_conditions", action="store_true", help="print the available condition names and exit")
    ap.add_argument("--severity", type=float, default=0.6, help="strength s of the corruptions (as --corruption_severity in training)")
    ap.add_argument("--light_knob", type=float, default=0.4,
                    help="poor_light / strong_light use the fixed brightness knob -/+ this value (as --lighting_knob_max)")
    ap.add_argument("--cond_seed", type=int, default=0, help="seed of the corruption (the same for every model)")
    ap.add_argument("--cond_title", action="store_true", help="write the condition name above each figure")
    ap.add_argument("--k", "--n", dest="k", type=int, default=5, help="number of palm images (columns)")
    ap.add_argument("--images", nargs="*", default=None, help="explicit image paths (takes priority over --image_dir)")
    ap.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="folder searched recursively when --images is not given")
    ap.add_argument("--pick", choices=["spread", "random"], default="spread",
                    help="how k images are taken from --image_dir: evenly spread over the sorted list, or random (--seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", choices=["last", "rollout"], default="last",
                    help="last = last layer only (default); rollout = attention rollout over all layers")
    ap.add_argument("--dino_query", choices=["cls", "patches"], default="cls",
                    help="DINOv2 rows only. cls (default) = what the CLS token attends to, which is what its "
                         "evaluation embedding uses; patches = mean over patch queries, comparable to the other rows")
    ap.add_argument("--out", default="attention_methods.png",
                    help="output base name; the condition is appended: attention_methods_<condition>.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_heads", type=int, default=None,
                    help="default: ContextEncoder's own rule max(4, embed_dim//32). The head count cannot be read "
                         "from the weights, so override this only if the models were trained with another value.")
    ap.add_argument("--alpha", type=float, default=0.55, help="opacity of the attention overlay")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--no_image_row", action="store_true", help="do not draw the row with the plain images")
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="multiplies every font size in the figure (row names, colorbar, entropy titles); e.g. 1.8")
    ap.add_argument("--label_size", type=float, default=None, help="row-name font size in pt (default 11 x --font_scale)")
    ap.add_argument("--title_size", type=float, default=None, help="entropy-title font size in pt (default 6 x --font_scale)")
    ap.add_argument("--cbar_size", type=float, default=None, help="colorbar font size in pt (default 8 x --font_scale)")
    ap.add_argument("--cell", type=float, default=2.15, help="height and width of one image cell in inches")
    ap.add_argument("--show_entropy", action="store_true", help="print H/Hmax (1 = uniform map) above every map")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    if a.list_conditions:
        print("conditions:", ", ".join(condition_names()))
        return
    if a.k < 1:
        raise SystemExit("--k must be >= 1")
    if a.font_scale <= 0 or a.cell <= 0:
        raise SystemExit("--font_scale and --cell must be > 0")
    if not a.images and not a.image_dir:
        raise SystemExit("Give --images ... or --image_dir ...")
    known = {_norm(n) for n in condition_names()}
    bad = [c for c in a.conditions if _norm(c) not in known]
    if bad:
        raise SystemExit(f"Unknown condition(s) {bad}. Available: {condition_names()}")

    specs = resolve_methods(a)
    paths = pick_images(a)
    pil = [Image.open(p).convert("RGB") for p in paths]
    print(f"\nimages ({len(paths)}):")
    for p in paths:
        print(f"  {p}")

    ms = load_models(specs, a)
    base, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    outs = []
    for ci, cond in enumerate(a.conditions):
        print(f"\n── condition: {cond} ──")
        rows = compute_rows(ms, pil, paths, cond, a, check=(ci == 0))
        out = f"{base}_{cond}{ext}"
        render_figure(rows, pil, paths, ms, cond, a, out)
        outs.append(out)
    print(f"\nsaved {len(outs)} figure(s): {', '.join(outs)}")


if __name__ == "__main__":
    main()
