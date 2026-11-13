"""visualize_predictor.py -- what does a trained JEPA-family model "know" about a MASKED palm region?

Palm-print version of the predictor visualisation of I-JEPA (Assran et al., CVPR 2023, Sec. 8 / App. E):
a stochastic decoder is trained in the style of RCDM (Bordes et al., 2022) to map a frozen representation
back to PIXELS. Everything the decoder draws consistently across noise seeds is information that the
representation contains; everything that changes from sample to sample is information it does not.

Recipe (as in the paper; every choice that is NOT in the paper is listed below):
  * the context encoder and the predictor are FROZEN (source=predictor);
  * for a palm image x, one context block and one target block are sampled (models.patchify);
  * s = the AVERAGE-POOLED predictor output for the target block (the predictor sees only the context
    tokens and the mask tokens placed at the target block);
  * the decoder h(x_t, t, s) is a conditional diffusion model (epsilon-prediction, linear beta schedule,
    1000 steps, 100 respaced sampling steps) that reconstructs the WHOLE image x from the noisy copy x_t and s;
    s is injected exactly as in RCDM's UNet: emb = time_embed(t) + Linear(s).
  * figure: image | context (target box in green) | n decoder samples per method (different noise seeds).

Differences from the paper / RCDM (be explicit about them in your write-up):
  * a compact UNet (base 64 channels, 4 resolution levels) written for this repo, NOT the 256-channel RCDM
    network, trained for --decoder_steps (default 30k) instead of 300k iterations; EMA 0.999 (RCDM: 0.9999);
  * fixed (posterior) sampling variance instead of RCDM's learned sigma;
  * the paper does not say which images the decoder was trained on; here: --train_images (recursive).

Two sources of the representation:
  --source predictor : paper-faithful. Needs the *_resume.pth file of each method (it is the only checkpoint that
                       contains the predictor; created by --save_ckpt <methods> with --save_resume 1, the default).
  --source encoder   : works with the plain ckpt_<tag>_train_<label>.pth files (target encoder only). s = the mean of the
                       TARGET-ENCODER tokens of the target block (--cond_region block) or of all tokens (--cond_region full,
                       i.e. the deployed embedding). This shows what the BACKBONE keeps, not what the predictor predicts.
  --source auto (default): predictor if every method has a *_resume.pth, otherwise encoder (printed).
The source is the same for all methods of one figure, so the rows are comparable.

Run from the repo root (models.py importable), e.g.
  python visualize_predictor.py --ckpt_dir out_xd_xpalm_xjtu_to_casia_aug8 \
      --train_images /home/pai-ng/Jamal/XJTU-UP /home/pai-ng/Jamal/xpalm \
      --image_dir /home/pai-ng/Jamal/CASIA-MS-ROI --k 3 --n_samples 4
Quick smoke test:  add  --decoder_steps 50 --sample_steps 10 --max_train_images 200
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from models import ContextEncoder, TargetEncoder, Predictor, patchify, _gather
from visualize_attention import load_state_dict, infer_arch

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_CKPT_DIR = "out_xd_xpalm_xjtu_to_casia_aug8"
DISPLAY_NAME = {"jepa": "JEPA", "dmtjepa": "DMT-JEPA", "palmjepa": "Palm-JEPA", "cjepa": "C-JEPA"}
DEFAULT_METHODS = ["jepa", "dmtjepa", "palmjepa"]
DEFAULT_TRG, DEFAULT_CTX = (0.10, 0.15), (0.90, 1.00)      # = config.py defaults


# ═══════════════════════════ images ═══════════════════════════
def list_images(root):
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                  if p.lower().endswith(IMG_EXTS))


def load_uint8(paths, size):
    """-> (N, 3, size, size) uint8. Resize = bilinear, like the dataset pipeline (transforms.Resize)."""
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
    return torch.stack(out)


def to_model(x_u8, device):
    """uint8 -> float in [-1, 1]; identical to ToTensor + Normalize([0.5]*3, [0.5]*3) of dataset.py."""
    return (x_u8.to(device).float() / 255.0 - 0.5) / 0.5


def pick_figure_images(args):
    if args.images:
        paths = list(args.images)
    else:
        found = list_images(args.image_dir)
        if len(found) < args.k:
            raise SystemExit(f"Only {len(found)} images under {args.image_dir}, need {args.k}.")
        if args.pick == "random":
            idx = sorted(np.random.RandomState(args.seed).choice(len(found), args.k, replace=False).tolist())
        else:
            idx = np.linspace(0, len(found) - 1, args.k).round().astype(int)
        paths = [found[i] for i in idx]
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Image not found: {p}")
    return paths


# ═══════════════════════════ checkpoints / methods ═══════════════════════════
def find_files(ckpt_dir, tag, train_label=None):
    """-> (plain files, resume files) of ckpt_<tag>_train_<label>[_resume].pth. The tag is the BASELINE_SPECS
    key exactly as written in the file name (underscores kept)."""
    allp = sorted(glob.glob(os.path.join(ckpt_dir, f"ckpt_{tag}_train_*.pth")))
    resume = [p for p in allp if p.endswith("_resume.pth")]
    plain = [p for p in allp if not p.endswith("_resume.pth")]
    if train_label:
        stem = f"ckpt_{tag}_train_{train_label}"
        resume = [p for p in resume if os.path.basename(p) == stem + "_resume.pth"]
        plain = [p for p in plain if os.path.basename(p) == stem + ".pth"]
    return plain, resume


def resolve_methods(args):
    """--methods items: 'tag' or 'tag=Row label'. -> list of dict(tag, label, plain, resume)."""
    out = []
    for item in args.methods:
        tag, label = (item.split("=", 1) + [None])[:2] if "=" in item else (item, None)
        tag = tag.strip()
        cdir = dict(kv.split("=", 1) for kv in (args.ckpt_dirs or [])).get(tag, args.ckpt_dir)
        plain, resume = find_files(cdir, tag, args.train_label)
        if not plain and not resume:
            print(f"  WARNING - skipped '{tag}': no ckpt_{tag}_train_*.pth in {cdir}")
            continue
        for kind, lst in (("plain", plain), ("resume", resume)):
            if len(lst) > 1:
                print(f"  [{tag}] {len(lst)} {kind} checkpoints match; using {os.path.basename(lst[0])} "
                      f"(choose another with --train_label)")
        out.append(dict(tag=tag, label=label or DISPLAY_NAME.get(tag.lower(), tag),
                        plain=plain[0] if plain else None, resume=resume[0] if resume else None))
    if not out:
        have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.ckpt_dir, "*.pth")))
        raise SystemExit(f"No checkpoint found. Files in {args.ckpt_dir}: {have}")
    return out


def load_resume(path):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)     # only load files you trust
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not (isinstance(blob, dict) and "modules" in blob):
        raise SystemExit(f"{path} is not a *_resume.pth written by ckpt_utils.Resumable")
    return blob


def build_predictor(sd, grid, D, cfgd):
    """Rebuild the predictor that was trained with the saved config: models.Predictor (JEPA / C-JEPA / Palm-JEPA)
    or dmtjepa_loss.DMTPredictor (DMT-JEPA). Returns (module, is_dmt, predict_structure)."""
    is_dmt = bool(cfgd.get("use_dmtjepa")) or "ctx_in_proj.weight" in sd
    if any(k.startswith("out_proj_struct.1.") for k in sd):
        norm_out = True                      # Sequential(Linear, LayerNorm)
    elif "out_proj_struct.weight" in sd:
        norm_out = False
    else:
        norm_out = True                      # unused (DMT without structure)
    if is_dmt:
        from dmtjepa_loss import DMTPredictor, LocalAggregationHead
        kw = {"use_structure": True} if "task_embed" in sd else {}
        pred = DMTPredictor(grid, D, context_agg_head=LocalAggregationHead(D), norm_struct_out=norm_out, **kw)
    else:
        pred = Predictor(grid, D, norm_struct_out=norm_out)
    try:
        pred.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise SystemExit(f"Predictor in the checkpoint does not match the rebuilt one:\n{e}")
    # During training the shared-trunk models ALWAYS called the predictor with predict_structure=True
    # (the structure token is part of the attention sequence); do the same here.
    use_a2 = bool(cfgd.get("use_a2")) or cfgd.get("struct_mode") in ("a2", "both")
    shared = bool(cfgd.get("use_shared_predictor_trunk", 1))
    predict_structure = bool(use_a2 and shared and "task_embed" in sd)
    return pred.eval(), is_dmt, predict_structure


def _freeze(m, device):
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class Method:
    """One trained method: .represent(x, ctx_masks, tgt_masks) -> (B, D) conditioning vector s."""

    def __init__(self, spec, source, cond_region, device):
        self.tag, self.label, self.source, self.cond_region = spec["tag"], spec["label"], source, cond_region
        self.cfgd = {}
        if source == "predictor":
            blob = load_resume(spec["resume"])
            mods, self.cfgd = blob["modules"], blob.get("config", {}) or {}
            for k in ("context_encoder", "predictor"):
                if k not in mods:
                    raise SystemExit(f"{spec['resume']}: module '{k}' missing (has {list(mods)})")
            a = infer_arch(mods["context_encoder"])
            S = a["patch"] * a["grid"]
            enc = ContextEncoder((S, S), a["grid"], a["embed_dim"], depth=a["depth"], num_heads=None,
                                 mlp_ratio=a["mlp_ratio"])
            enc.load_state_dict(mods["context_encoder"], strict=True)
            self.context_encoder = _freeze(enc, device)
            pred, self.is_dmt, self.predict_structure = build_predictor(
                mods["predictor"], a["grid"], a["embed_dim"], self.cfgd)
            self.predictor = _freeze(pred, device)
            self.epoch = blob.get("epoch")
        else:
            if spec["plain"]:
                sd = load_state_dict(spec["plain"])
            else:
                blob = load_resume(spec["resume"])
                if "target_encoder" not in blob["modules"]:
                    raise SystemExit(f"{spec['resume']}: no 'target_encoder' module")
                sd, self.cfgd = blob["modules"]["target_encoder"], blob.get("config", {}) or {}
            a = infer_arch(sd)
            S = a["patch"] * a["grid"]
            enc = TargetEncoder((S, S), a["grid"], a["embed_dim"], depth=a["depth"], num_heads=None,
                                mlp_ratio=a["mlp_ratio"])
            enc.load_state_dict(sd, strict=True)
            self.target_encoder = _freeze(enc, device)
            self.epoch = None
        self.arch, self.S, self.grid, self.D = a, S, a["grid"], a["embed_dim"]

    @torch.no_grad()
    def represent(self, x, ctx_masks, tgt_masks):
        B = x.size(0)
        if self.source == "predictor":
            ctx = self.context_encoder(x, ctx_masks)                         # (B, Nc, D): only context tokens
            if self.is_dmt:
                preds = (self.predictor(ctx, tgt_masks, predict_structure=True)[0] if self.predict_structure
                         else self.predictor(ctx, tgt_masks))
                out = preds[0]                                                # list over target blocks
            else:
                out = (self.predictor(ctx, ctx_masks, tgt_masks, predict_structure=True)[0]
                       if self.predict_structure else self.predictor(ctx, ctx_masks, tgt_masks))
                out = out[:B]                                                 # first (= only) target block
            return out.mean(dim=1)                                            # average-pooled predictor output
        z = self.target_encoder(x)                                            # (B, P, D), clean full image
        if self.cond_region == "full":
            return z.mean(dim=1)
        return _gather(z, tgt_masks[0]).mean(dim=1)


def mask_ratios(args, methods):
    """CLI > the config stored in the resume checkpoints > config.py defaults."""
    trg = tuple(args.trg_ratio) if args.trg_ratio else None
    ctx = tuple(args.ctx_ratio) if args.ctx_ratio else None
    for m in methods:
        if trg is None and m.cfgd.get("trg_ratio"):
            trg = tuple(m.cfgd["trg_ratio"])
        if ctx is None and m.cfgd.get("ctx_ratio"):
            ctx = tuple(m.cfgd["ctx_ratio"])
    return trg or DEFAULT_TRG, ctx or DEFAULT_CTX


# ═══════════════════════════ RCDM-style conditional diffusion decoder ═══════════════════════════
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def gn(ch):
    return nn.GroupNorm(math.gcd(32, ch), ch)


def zero_(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class ResBlock(nn.Module):
    """guided-diffusion ResBlock with scale-shift norm (RCDM default)."""

    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.in_norm, self.in_conv = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * cout))
        self.out_norm, self.drop = gn(cout), nn.Dropout(dropout)
        self.out_conv = zero_(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x, emb):
        h = self.in_conv(F.silu(self.in_norm(x)))
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_conv(self.drop(F.silu(h)))


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        assert ch % heads == 0, (ch, heads)
        self.heads = heads
        self.norm, self.qkv = gn(ch), nn.Conv1d(ch, 3 * ch, 1)
        self.proj = zero_(nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W)
        q, k, v = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)   # (B, heads, d, N)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        o = o.transpose(-1, -2).reshape(B, C, H * W)
        return x + self.proj(o).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


def _run(blocks, h, emb):
    for b in blocks:
        h = b(h, emb) if isinstance(b, ResBlock) else b(h)
    return h


class CondUNet(nn.Module):
    """eps_theta(x_t, t, s). The representation s enters exactly like RCDM's UNetModel:
         emb = time_embed(timestep_embedding(t)) + Linear(s)        (the sum goes to every ResBlock)."""

    def __init__(self, feat_dim, base=64, mult=(1, 2, 3, 4), n_res=2, attn_levels=(3,), heads=4, dropout=0.0):
        super().__init__()
        self.base, self.factor = base, 2 ** (len(mult) - 1)
        emb_dim = base * 4
        self.time_embed = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.ssl_emb = nn.Linear(feat_dim, emb_dim)
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.downs, chans, ch = nn.ModuleList(), [base], base
        for lvl, m in enumerate(mult):
            for _ in range(n_res):
                blocks = [ResBlock(ch, base * m, emb_dim, dropout)]
                ch = base * m
                if lvl in attn_levels:
                    blocks.append(AttnBlock(ch, heads))
                self.downs.append(nn.ModuleList(blocks)); chans.append(ch)
            if lvl != len(mult) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)])); chans.append(ch)
        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch, heads),
                                  ResBlock(ch, ch, emb_dim, dropout)])
        self.ups = nn.ModuleList()
        for lvl, m in list(enumerate(mult))[::-1]:
            for i in range(n_res + 1):
                skip = chans.pop()
                blocks = [ResBlock(ch + skip, base * m, emb_dim, dropout)]
                ch = base * m
                if lvl in attn_levels:
                    blocks.append(AttnBlock(ch, heads))
                if lvl and i == n_res:
                    blocks.append(Upsample(ch))
                self.ups.append(nn.ModuleList(blocks))
        self.out = nn.Sequential(gn(ch), nn.SiLU(), zero_(nn.Conv2d(ch, 3, 3, padding=1)))

    def forward(self, x, t, feat):
        assert x.shape[-1] % self.factor == 0 and x.shape[-2] % self.factor == 0, \
            f"decoder size must be divisible by {self.factor}, got {tuple(x.shape[-2:])}"
        emb = self.time_embed(timestep_embedding(t, self.base)) + self.ssl_emb(feat)
        h = self.stem(x)
        hs = [h]
        for blocks in self.downs:
            h = _run(blocks, h, emb)
            hs.append(h)
        h = _run(self.mid, h, emb)
        for blocks in self.ups:
            h = _run(blocks, torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)


class Diffusion:
    """DDPM, epsilon-prediction, linear beta schedule (1e-4 .. 0.02, T = 1000) -- RCDM / guided-diffusion defaults."""

    def __init__(self, device, T=1000):
        self.T, self.device = T, device
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
        self.ac = torch.cumprod(1.0 - betas, dim=0)                           # alpha_bar, float64

    def _g(self, a, t):
        return a.to(self.device)[t].float()[:, None, None, None]

    def loss(self, net, x0, feat):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self._g(self.ac.sqrt(), t) * x0 + self._g((1 - self.ac).sqrt(), t) * noise
        return F.mse_loss(net(xt, t, feat), noise)

    @torch.no_grad()
    def sample(self, net, feat, size, steps=100, seed=0, clip=True):
        """Ancestral sampling over `steps` evenly respaced timesteps (RCDM: --timestep_respacing 100).
        feat: (n, D) -> (n, 3, size, size) in [-1, 1]. Same seed => same initial noise and same step noise."""
        n = feat.size(0)
        use = np.unique(np.linspace(0, self.T - 1, steps).round().astype(int))
        ac = self.ac[torch.as_tensor(use)]
        ac_prev = torch.cat([torch.ones(1, dtype=ac.dtype), ac[:-1]])
        beta = 1.0 - ac / ac_prev
        post_var = beta * (1.0 - ac_prev) / (1.0 - ac)
        c1 = beta * ac_prev.sqrt() / (1.0 - ac)
        c2 = (1.0 - ac_prev) * (1.0 - beta).sqrt() / (1.0 - ac)
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn(n, 3, size, size, device=self.device, generator=g)
        for i in reversed(range(len(use))):
            t = torch.full((n,), int(use[i]), device=self.device, dtype=torch.long)
            eps = net(x, t, feat)
            x0 = (x - float((1 - ac[i]).sqrt()) * eps) / float(ac[i].sqrt())
            if clip:
                x0 = x0.clamp(-1, 1)
            mean = float(c1[i]) * x0 + float(c2[i]) * x
            if i > 0:
                x = mean + float(post_var[i].sqrt()) * torch.randn(x.shape, device=self.device, generator=g)
            else:
                x = mean
        return x


# ═══════════════════════════ progress signals ═══════════════════════════
def _fork_rng(device):
    d = torch.device(device)
    return torch.random.fork_rng(devices=[d.index or 0] if d.type == "cuda" else [])   # keep the training RNG untouched


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def make_val_set(m, val_u8, trg, ctx, device, S_dec):
    """FIXED held-out batch (images never used for decoder training): masks, representation s, timesteps, noise."""
    g = torch.Generator().manual_seed(1234)
    x = to_model(val_u8, device)
    with _fork_rng(device):
        torch.manual_seed(1234)
        ctx_masks, tgt_masks = patchify(x.size(0), m.grid, 1, trg_ratio=trg, ctx_ratio=ctx, device=device)
        feat = m.represent(x, ctx_masks, tgt_masks)
    if S_dec != m.S:
        x = F.interpolate(x, size=(S_dec, S_dec), mode="bilinear", align_corners=False, antialias=True)
    t = torch.randint(0, 1000, (x.size(0),), generator=g).to(device)
    noise = torch.randn(x.shape, generator=g).to(device)
    return dict(x=x, feat=feat, t=t, noise=noise)


@torch.no_grad()
def val_losses(net, diff, V, chunk=64):
    """eps-prediction MSE on the fixed held-out batch with the TRUE s and with s taken from ANOTHER image.
    gain = loss(shuffled s) - loss(true s) > 0 means the decoder really uses the representation; ~0 means it ignores it.
    gain_hiT is the same restricted to the noisiest third of the timesteps (t >= 667), where s matters most."""
    xt = diff._g(diff.ac.sqrt(), V["t"]) * V["x"] + diff._g((1 - diff.ac).sqrt(), V["t"]) * V["noise"]
    shuf = V["feat"].roll(1, 0)
    e_true, e_shuf = [], []
    for i in range(0, xt.size(0), chunk):
        sl = slice(i, i + chunk)
        for feat, acc in ((V["feat"], e_true), (shuf, e_shuf)):
            acc.append(((net(xt[sl], V["t"][sl], feat[sl]) - V["noise"][sl]) ** 2).mean(dim=(1, 2, 3)))
    e_true, e_shuf = torch.cat(e_true), torch.cat(e_shuf)
    hi = V["t"] >= 667
    return dict(val=e_true.mean().item(), val_shuf=e_shuf.mean().item(), gain=(e_shuf - e_true).mean().item(),
                gain_hiT=(e_shuf - e_true)[hi].mean().item() if bool(hi.any()) else float("nan"))


@torch.no_grad()
def make_preview(args, m, net, prev_u8, trg, ctx, device, S_dec, step):
    """Small figure (Image | Context | n samples) of the current EMA decoder on FIXED images and masks."""
    diff, K, n, patch = Diffusion(device), prev_u8.size(0), args.n_samples, m.S // m.grid
    fig, ax = plt.subplots(K, 2 + n, figsize=(1.45 * (2 + n), 1.5 * K + 0.5), squeeze=False)
    for i in range(K):
        img = prev_u8[i].permute(1, 2, 0).numpy()
        with _fork_rng(device):
            torch.manual_seed(args.seed * 1000 + i)                       # same masks as the final figure
            ctx_masks, tgt_masks = patchify(1, m.grid, 1, trg_ratio=trg, ctx_ratio=ctx, device=device)
        bx = box_of(tgt_masks[0][0].cpu().numpy(), m.grid, patch)
        s = m.represent(to_model(prev_u8[i:i + 1], device), ctx_masks, tgt_masks)
        out = diff.sample(net, s.repeat(n, 1), S_dec, steps=args.preview_steps, seed=args.seed * 1000 + i)
        if S_dec != m.S:
            out = F.interpolate(out, size=(m.S, m.S), mode="bilinear", align_corners=False)
        out = ((out.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(context_image(img, ctx_masks[0][0].cpu().numpy(), m.grid, patch))
        ax[i, 1].add_patch(Rectangle(bx[:2], bx[2], bx[3], fill=False, ec="#00c000", lw=1.6))
        for j in range(n):
            ax[i, 2 + j].imshow(out[j])
            ax[i, 2 + j].add_patch(Rectangle(bx[:2], bx[2], bx[3], fill=False, ec="#00c000", lw=1.2))
    for a_ in ax.ravel():
        a_.axis("off")
    fig.suptitle(f"{m.label}: EMA decoder at step {step}", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.03, hspace=0.04)
    f1 = os.path.join(args.out_dir, f"preview_{m.tag}_step{step:06d}.png")
    fig.savefig(f1, dpi=110, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(args.out_dir, f"preview_{m.tag}_latest.png"), dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return f1


def save_curves(path, hist):
    if len(hist) < 2:
        return
    st = [h["step"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].plot(st, [h["loss"] for h in hist], label="train (running avg)")
    ax[0].plot(st, [h["val"] for h in hist], label="held-out (EMA net)")
    ax[0].plot(st, [h["val_shuf"] for h in hist], "--", label="held-out, s shuffled")
    ax[0].set_title("eps-MSE"); ax[0].legend(fontsize=7)
    ax[1].plot(st, [h["gain"] for h in hist], label="all t")
    ax[1].plot(st, [h["gain_hiT"] for h in hist], label="noisiest third")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_title("gain = loss(shuffled s) - loss(true s)"); ax[1].legend(fontsize=7)
    ax[2].plot(st, [h["gnorm"] for h in hist]); ax[2].set_title("grad norm (before clipping)")
    for a_ in ax:
        a_.set_xlabel("step"); a_.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ═══════════════════════════ decoder training ═══════════════════════════
def decoder_path(args, m):
    name = f"decoder_{m.tag}_{m.source}" + (f"_{m.cond_region}" if m.source == "encoder" else "") + ".pth"
    return os.path.join(args.out_dir, name)


def ema_path(args, m):
    return decoder_path(args, m).replace(".pth", "_ema.pth")


def make_net(args, m, device):
    return CondUNet(m.D, base=args.decoder_ch, heads=args.decoder_heads).to(device)


def write_ema(args, m, ema, step, S_dec):
    """Small inference file: EMA weights + everything needed to rebuild the net (no optimizer)."""
    p = ema_path(args, m)
    if os.path.isfile(p):
        try:
            if int(torch.load(p, map_location="cpu").get("step", -1)) == step:
                return
        except Exception:
            pass
    torch.save({"ema": ema.state_dict(), "step": step, "S_dec": S_dec, "feat_dim": m.D, "base": args.decoder_ch,
                "heads": args.decoder_heads, "tag": m.tag, "source": m.source, "cond_region": m.cond_region,
                "label": m.label, "sample_steps": args.sample_steps}, p)


def train_decoder(args, m, train_u8, trg, ctx, device, preview_u8=None):
    path = decoder_path(args, m)
    S_dec = args.decoder_size or m.S
    net = make_net(args, m, device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    step0 = 0
    if os.path.isfile(path) and not args.retrain:
        ck = torch.load(path, map_location="cpu")
        if ck.get("S_dec") == S_dec and ck.get("feat_dim") == m.D and ck.get("base") == args.decoder_ch:
            net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
            step0 = int(ck["step"])
            print(f"  [{m.label}] found {path} (step {step0})")
        else:
            print(f"  [{m.label}] {path} has a different size/width -> training from scratch")
    elif os.path.isfile(ema_path(args, m)) and not args.retrain:       # inference-only file (no optimizer): cannot resume
        ck = torch.load(ema_path(args, m), map_location="cpu")
        if ck.get("S_dec") == S_dec and ck.get("feat_dim") == m.D and ck.get("base") == args.decoder_ch:
            ema.load_state_dict(ck["ema"]); step0 = int(ck["step"])
            if step0 < args.decoder_steps:
                raise SystemExit(f"[{m.label}] only the EMA-only file exists (step {step0}); it cannot be trained further. "
                                 f"Lower --decoder_steps to {step0} or use the full decoder_*.pth.")
            print(f"  [{m.label}] using EMA-only file {ema_path(args, m)} (step {step0})")
    if step0 >= args.decoder_steps:
        write_ema(args, m, ema, step0, S_dec)                            # make sure the inference file exists
        return ema
    if train_u8 is None:
        raise SystemExit(f"[{m.label}] decoder needs training (step {step0} < {args.decoder_steps}): give --train_images")
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  [{m.label}] training decoder: {n_par:.1f}M params, {args.decoder_steps - step0} steps, "
          f"batch {args.batch_size} x accum {args.accum} = {args.batch_size * args.accum}, "
          f"{'bf16 autocast, ' if args.amp else ''}size {S_dec}, source={m.source}")
    diff, N, run, gnorm = Diffusion(device), train_u8.size(0), 0.0, 0.0
    nval = min(args.val_images, N // 10)                                 # held-out images for the validation signals
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    val_idx, tr_idx = perm[:nval], perm[nval:]
    V = make_val_set(m, train_u8[val_idx], trg, ctx, device, S_dec) if nval > 0 else None
    print(f"  [{m.label}] {len(tr_idx)} training / {nval} held-out images; validation every {args.val_every} steps, "
          f"preview every {args.preview_every} steps -> {args.out_dir}")
    csv_path = os.path.join(args.out_dir, f"train_log_{m.tag}.csv")
    hist, t0 = [], time.time()
    net.train()

    def save(step):
        torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "S_dec": S_dec, "feat_dim": m.D, "base": args.decoder_ch, "tag": m.tag,
                    "source": m.source, "cond_region": m.cond_region}, path)

    for step in range(step0 + 1, args.decoder_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(args.accum):                                   # gradient accumulation: effective batch = batch_size * accum
            idx = tr_idx[torch.randint(0, len(tr_idx), (args.batch_size,))]
            x = to_model(train_u8[idx], device)
            ctx_masks, tgt_masks = patchify(x.size(0), m.grid, 1, trg_ratio=trg, ctx_ratio=ctx, device=device)
            feat = m.represent(x, ctx_masks, tgt_masks)
            if S_dec != m.S:
                x = F.interpolate(x, size=(S_dec, S_dec), mode="bilinear", align_corners=False, antialias=True)
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=args.amp):
                loss = diff.loss(net, x, feat) / args.accum
            loss.backward()
            loss_val += loss.item()
        gn_ = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(args.ema).add_(p.detach(), alpha=1.0 - args.ema)
        first = step == step0 + 1
        run = loss_val if first else 0.98 * run + 0.02 * loss_val
        gnorm = float(gn_) if first else 0.98 * gnorm + 0.02 * float(gn_)
        if step % args.log_every == 0 or step == args.decoder_steps:
            el = time.time() - t0
            ips = (step - step0) / max(el, 1e-9)
            mem = f" | {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if str(device).startswith("cuda") else ""
            print(f"    [{m.label}] step {step}/{args.decoder_steps} | loss {run:.4f} | gnorm {gnorm:.2f} | "
                  f"{ips:.2f} it/s | {hms(el)} elapsed, ETA {hms((args.decoder_steps - step) / ips)}{mem}", flush=True)
        if V is not None and (step % args.val_every == 0 or step == args.decoder_steps):
            vl = val_losses(ema, diff, V)
            ips = (step - step0) / max(time.time() - t0, 1e-9)
            print(f"      held-out: loss {vl['val']:.4f} | with s shuffled {vl['val_shuf']:.4f} | gain {vl['gain']:+.4f} "
                  f"(noisiest third {vl['gain_hiT']:+.4f})", flush=True)
            hist.append(dict(step=step, loss=run, gnorm=gnorm, **vl))
            new = not os.path.isfile(csv_path)
            with open(csv_path, "a") as f:
                if new:
                    f.write("step,train_loss,gnorm,val_loss,val_loss_shuffled_s,gain,gain_hiT,it_per_s\n")
                f.write(f"{step},{run:.5f},{gnorm:.4f},{vl['val']:.5f},{vl['val_shuf']:.5f},{vl['gain']:.5f},"
                        f"{vl['gain_hiT']:.5f},{ips:.3f}\n")
            save_curves(os.path.join(args.out_dir, f"train_curves_{m.tag}.png"), hist)
        if preview_u8 is not None and args.preview_every > 0 and (step % args.preview_every == 0 or step == args.decoder_steps):
            print(f"      preview -> {make_preview(args, m, ema, preview_u8, trg, ctx, device, S_dec, step)}", flush=True)
        if step % args.save_every == 0 and step != args.decoder_steps:
            save(step)
        if args.snapshot_every > 0 and step % args.snapshot_every == 0:
            write_ema(args, m, ema, step, S_dec)
    save(args.decoder_steps)
    write_ema(args, m, ema, args.decoder_steps, S_dec)
    print(f"  [{m.label}] saved {path}")
    return ema.eval()


# ═══════════════════════════ figure ═══════════════════════════
def box_of(idx, grid, patch):
    """pixel rectangle (x, y, w, h), imshow coordinates, of a rectangular patch block given by flat indices."""
    idx = np.asarray(idx)
    r, c = idx // grid, idx % grid
    return (c.min() * patch - 0.5, r.min() * patch - 0.5,
            (c.max() - c.min() + 1) * patch, (r.max() - r.min() + 1) * patch)


def context_image(img_u8, ctx_idx, grid, patch):
    vis = np.zeros(grid * grid, dtype=bool)
    vis[np.asarray(ctx_idx)] = True
    up = np.kron(vis.reshape(grid, grid), np.ones((patch, patch), dtype=bool))
    out = img_u8.copy()
    out[~up] = 0                                                    # patches the context encoder never sees
    return out


def main():
    ap = argparse.ArgumentParser(description="Decoder visualisation of the predictor / encoder representation (RCDM style)")
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--ckpt_dirs", nargs="*", default=None, metavar="TAG=DIR",
                    help="per-method checkpoint folder overriding --ckpt_dir, e.g. palmjepa=/path/a jepa=/path/b")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help="rows, top to bottom: baseline keys as in the file names (jepa dmtjepa palmjepa ...), "
                         "optionally 'key=Row label'")
    ap.add_argument("--train_label", default=None, help="only if a method has several checkpoints, e.g. 'xpalm+xjtu'")
    ap.add_argument("--source", choices=["auto", "predictor", "encoder"], default="auto")
    ap.add_argument("--cond_region", choices=["block", "full"], default="block",
                    help="source=encoder only: mean of the target-encoder tokens of the target block, or of all tokens")
    ap.add_argument("--train_images", nargs="*", default=None,
                    help="folder(s), searched recursively, whose images train the decoders")
    ap.add_argument("--max_train_images", type=int, default=20000)
    ap.add_argument("--image_dir", default="/home/pai-ng/Jamal/CASIA-MS-ROI", help="figure images (recursive)")
    ap.add_argument("--images", nargs="*", default=None, help="explicit figure images (priority over --image_dir)")
    ap.add_argument("--k", type=int, default=3, help="number of figure images (rows)")
    ap.add_argument("--pick", choices=["spread", "random"], default="spread")
    ap.add_argument("--n_samples", type=int, default=4, help="decoder samples per method and image")
    ap.add_argument("--trg_ratio", type=float, nargs=2, default=None, help="default: from the checkpoints, else 0.10 0.15")
    ap.add_argument("--ctx_ratio", type=float, nargs=2, default=None, help="default: from the checkpoints, else 0.90 1.00")
    ap.add_argument("--out_dir", default="predictor_viz", help="decoders (*.pth) and the figure are written here")
    ap.add_argument("--out", default="predictor_viz.png", help="figure file name inside --out_dir")
    ap.add_argument("--decoder_steps", type=int, default=30000)
    ap.add_argument("--decoder_ch", type=int, default=64)
    ap.add_argument("--decoder_heads", type=int, default=4)
    ap.add_argument("--decoder_size", type=int, default=0, help="0 = the model's image size; must be divisible by 8")
    ap.add_argument("--batch_size", type=int, default=32, help="images per forward/backward pass (lower it on CUDA OOM)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation passes per update; effective batch = batch_size * accum")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the decoder training (about half the activation memory)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument("--retrain", action="store_true", help="ignore existing decoder files")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=5000, help="full training checkpoint (with optimizer) every N steps")
    ap.add_argument("--snapshot_every", type=int, default=5000,
                    help="small EMA-only inference file (decoder_<tag>_<source>_ema.pth) every N steps (0 = only at the end)")
    ap.add_argument("--val_every", type=int, default=1000, help="held-out loss / shuffled-s gain every N steps")
    ap.add_argument("--val_images", type=int, default=256, help="training images held out for the validation signals (0 = off)")
    ap.add_argument("--preview_every", type=int, default=5000, help="small preview figure every N steps (0 = off)")
    ap.add_argument("--preview_steps", type=int, default=50, help="sampling steps of the previews (faster than --sample_steps)")
    ap.add_argument("--preview_k", type=int, default=3, help="number of figure images in the previews")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    device = a.device
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    # ── which methods / which representation ──
    specs = resolve_methods(a)
    miss = [s["tag"] for s in specs if not s["resume"]]
    if a.source == "auto":
        source = "predictor" if not miss else "encoder"
        print(f"\nsource = {source}" + (f"  (no *_resume.pth for: {miss} -> falling back to the target-encoder "
                                        f"representation; re-run training with --save_ckpt <keys> to get predictor files)"
                                        if miss else ""))
    else:
        source = a.source
        if source == "predictor" and miss:
            raise SystemExit(f"--source predictor needs ckpt_<key>_train_<label>_resume.pth, missing for {miss}. They are "
                             f"written when training with --save_ckpt <keys> (and --save_resume 1, the default). "
                             f"Use --source encoder to work with the plain checkpoints.")
    methods = [Method(s, source, a.cond_region, device) for s in specs]
    S, grid = methods[0].S, methods[0].grid
    for m in methods:
        if (m.S, m.grid) != (S, grid):
            raise SystemExit(f"methods differ in image size / patch grid: {m.label} {(m.S, m.grid)} vs {(S, grid)}")
        print(f"  {m.label:<12} D={m.D} grid={grid}x{grid} image={S}px" +
              (f" | predictor: {'DMT' if m.is_dmt else 'standard'}, structure token {'ON' if m.predict_structure else 'off'}"
               f", saved at epoch {m.epoch}" if source == "predictor" else f" | target encoder, region={a.cond_region}"))
    patch = S // grid
    S_dec = a.decoder_size or S
    if S_dec % 8:
        raise SystemExit("--decoder_size must be divisible by 8")
    trg, ctx = mask_ratios(a, methods)
    print(f"  masks: target block ratio {trg}, context block ratio {ctx}")

    # ── training images ──
    train_u8 = None
    if a.train_images:
        paths = [p for d in a.train_images for p in list_images(d)]
        if not paths:
            raise SystemExit(f"no images under {a.train_images}")
        if len(paths) > a.max_train_images:
            sel = np.random.RandomState(a.seed).choice(len(paths), a.max_train_images, replace=False)
            paths = [paths[i] for i in sorted(sel)]
        print(f"\ndecoder training images: {len(paths)}")
        train_u8 = load_uint8(paths, S)

    # ── figure images (also used for the previews while the decoders train) ──
    fig_paths = pick_figure_images(a)
    print("\nfigure images:")
    for p in fig_paths:
        print(f"  {p}")
    imgs_u8 = load_uint8(fig_paths, S)

    # ── decoders ──
    print()
    decoders = {m.tag: train_decoder(a, m, train_u8, trg, ctx, device, preview_u8=imgs_u8[:a.preview_k]) for m in methods}
    del train_u8

    # ── figure ──
    diff = Diffusion(device)
    K, M, n = len(fig_paths), len(methods), a.n_samples
    cols = 2 + M * n
    fig, ax = plt.subplots(K, cols, figsize=(1.45 * cols, 1.5 * K + 0.5), squeeze=False)
    rec = dict(images=imgs_u8.permute(0, 2, 3, 1).numpy(), contexts=[], boxes=[], samples=np.zeros((M, K, n, S, S, 3), np.uint8))
    for i in range(K):
        img = imgs_u8[i].permute(1, 2, 0).numpy()
        torch.manual_seed(a.seed * 1000 + i)                          # same masks for every method
        ctx_masks, tgt_masks = patchify(1, grid, 1, trg_ratio=trg, ctx_ratio=ctx, device=device)
        bx = box_of(tgt_masks[0][0].cpu().numpy(), grid, patch)
        x = to_model(imgs_u8[i:i + 1], device)
        ax[i, 0].imshow(img)
        cimg = context_image(img, ctx_masks[0][0].cpu().numpy(), grid, patch)
        rec["contexts"].append(np.asarray(cimg)); rec["boxes"].append(np.asarray(bx, dtype=float))
        ax[i, 1].imshow(cimg)
        ax[i, 1].add_patch(Rectangle(bx[:2], bx[2], bx[3], fill=False, ec="#00c000", lw=1.6))
        for m_i, m in enumerate(methods):
            s = m.represent(x, ctx_masks, tgt_masks)                  # (1, D)
            out = diff.sample(decoders[m.tag], s.repeat(n, 1), S_dec, steps=a.sample_steps, seed=a.seed * 1000 + i)
            if S_dec != S:
                out = F.interpolate(out, size=(S, S), mode="bilinear", align_corners=False)
            out = ((out.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
            rec["samples"][m_i, i] = (out * 255 + 0.5).astype(np.uint8)
            for j in range(n):
                a_ = ax[i, 2 + m_i * n + j]
                a_.imshow(out[j])
                a_.add_patch(Rectangle(bx[:2], bx[2], bx[3], fill=False, ec="#00c000", lw=1.2))
        print(f"  image {i + 1}/{K} done")
    for a_ in ax.ravel():
        a_.axis("off")
    ax[0, 0].set_title("Image", fontsize=9)
    ax[0, 1].set_title("Context", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90 if K > 1 else 0.86, bottom=0.01, wspace=0.03, hspace=0.04)
    for m_i, m in enumerate(methods):
        p0, p1 = ax[0, 2 + m_i * n].get_position(), ax[0, 2 + m_i * n + n - 1].get_position()
        fig.text((p0.x0 + p1.x1) / 2, p0.y1 + 0.012, m.label, ha="center", va="bottom", fontsize=11, fontweight="bold")
    out_png = os.path.join(a.out_dir, a.out)
    fig.savefig(out_png, dpi=a.dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"\nsaved {out_png} (+ .pdf)")
    base = os.path.splitext(out_png)[0]
    np.savez_compressed(base + "_samples.npz", images=rec["images"], contexts=np.stack(rec["contexts"]),
                        boxes=np.stack(rec["boxes"]), samples=rec["samples"], paths=np.array(fig_paths),
                        methods=np.array([m.tag for m in methods]), labels=np.array([m.label for m in methods]))
    with open(base + "_config.json", "w") as f:
        json.dump(dict(args=vars(a), source=source, S=S, grid=grid, trg_ratio=trg, ctx_ratio=ctx,
                       checkpoints={s["tag"]: dict(plain=s["plain"], resume=s["resume"]) for s in specs},
                       decoders={m.tag: os.path.basename(ema_path(a, m)) for m in methods}), f, indent=1, default=str)
    print(f"saved {base}_samples.npz and {base}_config.json (re-draw the figure without the decoders)")


if __name__ == "__main__":
    main()
