"""
analyze_domain_distribution.py -- standalone motivation-figure generator.

Trains a plain JEPA (or any --method jepa variant, via METHOD_EXTRA_FLAGS
below) on selected training domains, then produces two figures:

  (B) Genuine/impostor cosine-similarity distributions across 4 modes:
      1. seen_dom_seen_id    -- the TRAINING samples themselves, no
                                 resplitting (train_loader.dataset.samples
                                 fed straight into split_gallery_probe).
      2. seen_dom_unseen_id  -- reuses build_datasets()'s own eval_dict.
      3. unseen_dom_seen_id  -- reuses build_datasets()'s own eval_dict.
      4. unseen_dom_unseen_id -- reuses build_datasets()'s own eval_dict.

  (C) t-SNE/UMAP of pooled samples from ALL domains of the training
      dataset PLUS all other datasets (CASIA-MS/XJTU-UP/X-Palm, whichever
      aren't the training one), colored by domain (or by source dataset),
      under a stratified point cap.

No config.py/ci_utils.py integration -- everything is set in the
PARAMETERS block below and this file is run directly:
    python analyze_domain_distribution.py

Reuses (does not duplicate) dataset.py's build_datasets/split_gallery_
_probe/scan_by_key/build_id_map, evaluate.py's extract_features, and
models.py's ContextEncoder/TargetEncoder/Predictor/patchify/apply_masks/
repeat_interleave_batch/update_ema -- the exact same training mechanics
main.py's train_jepa() uses, stripped down (no CI, no cross-dataset EER
table, no output-file writing) since this script's job is figures, not
another results table.
"""

import os
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_cfg
from dataset import (build_datasets, split_gallery_probe, build_id_map,
                      scan_by_key, normalize_dataset_key, CASIADataset)
from models import (ContextEncoder, TargetEncoder, Predictor,
                     FeatureExtractor, patchify, apply_masks,
                     repeat_interleave_batch, update_ema)
from evaluate import extract_features
from torch.utils.data import DataLoader

if XPALM_DEVICE_FILTER:
    import dataset as _dataset_module
    _original_scan_xpalm = _dataset_module.scan_xpalm

    def _filtered_scan_xpalm(data_root):
        samples = _original_scan_xpalm(data_root)
        filtered = [s for s in samples if s.get("device") == XPALM_DEVICE_FILTER]
        print(f"  [X-Palm] device filter='{XPALM_DEVICE_FILTER}': "
              f"{len(filtered)}/{len(samples)} samples kept")
        return filtered

    _dataset_module.scan_xpalm = _filtered_scan_xpalm



# ═══════════════════════════════════════════════════════════════
#  PARAMETERS -- edit these, then just run the script directly
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "/home/pai-ng/Jamal/xpalm"
TRAIN_SPECTRUMS = ["sf", "close", "jf", "fl", "bf", "rnd"]
TEST_SPECTRUMS = None            # None = every other SMARTPHONE domain
                                  # (scanner excluded via XPALM_DEVICE_FILTER below)
MODE = "cross_domain_openset"

CASIA_DIR = "/home/pai-ng/Jamal/CASIA-MS-ROI"
XJTU_DIR = "/home/pai-ng/Jamal/XJTU-UP"
XPALM_DIR = "/home/pai-ng/Jamal/xpalm"

# X-Palm-specific: scan_xpalm() tags every sample with "device": "scanner"
# or "smartphone" -- restrict to smartphone-only here (affects BOTH the
# train/eval split via build_datasets AND the Option C t-SNE pool).
XPALM_DEVICE_FILTER = "smartphone"   # "smartphone", "scanner", or None (both)

# Any --method jepa flag combo works here -- plain JEPA for now.
# To try Palm-JEPA instead, e.g.:
#   METHOD_EXTRA_FLAGS = {"use_corruption": 1, "struct_mode": "a2",
#                          "struct_loss": "infonce", "w_a2": 0.3}
METHOD_EXTRA_FLAGS = {"use_corruption": 0}

EPOCHS = 200
EMBED_DIM = 256
NUM_PATCHES = 8
BATCH_SIZE = 64
GALLERY_RATIO = 0.5

OUTPUT_DIR = "./out_domain_analysis_xpalm"
MAX_TSNE_POINTS = 3000
REDUCTION_METHOD = "tsne"        # "tsne" or "umap"
COLOR_BY = "spectrum"            # "spectrum" (fine domain) or "dataset"
SEED = 2025
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cfg():
    """Build a cfg object via the real config.py parser, so every field
    downstream code expects (cfg.img_size, cfg.num_blocks, cfg.trg_ratio,
    etc.) is populated with real defaults -- not a hand-rolled stand-in
    that could silently diverge from what train_jepa actually assumes."""
    args = [
        "--data_dir", DATA_DIR,
        "--mode", MODE,
        "--train_spectrums", *TRAIN_SPECTRUMS,
        "--embed_dim", str(EMBED_DIM),
        "--num_patches", str(NUM_PATCHES),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--gallery_ratio", str(GALLERY_RATIO),
        "--seed", str(SEED),
        "--device", DEVICE,
        "--output_dir", OUTPUT_DIR,
        "--method", "jepa",
    ]
    if TEST_SPECTRUMS:
        args += ["--test_spectrums", *TEST_SPECTRUMS]
    for k, v in METHOD_EXTRA_FLAGS.items():
        args += [f"--{k}", str(v)]
    return get_cfg(args)


# ═══════════════════════════════════════════════════════════════
#  Step 1: train a plain JEPA (stripped-down copy of train_jepa's core)
# ═══════════════════════════════════════════════════════════════

def train_plain_jepa(cfg, train_loader):
    img_size = (cfg.img_size, cfg.img_size)
    print(f"\n Building JEPA (analysis run)...")
    context_encoder = ContextEncoder(img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    target_encoder = TargetEncoder(img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    predictor = Predictor(cfg.num_patches, cfg.embed_dim,
                          norm_struct_out=bool(cfg.norm_struct_out)).to(cfg.device)

    for pc, pt in zip(context_encoder.parameters(), target_encoder.parameters()):
        pt.data.copy_(pc.data)
    for p in target_encoder.parameters():
        p.requires_grad = False

    train_params = list(context_encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(train_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return cfg.start_lr / cfg.learning_rate + \
                (1 - cfg.start_lr / cfg.learning_rate) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg.final_lr / cfg.learning_rate + \
            (1 - cfg.final_lr / cfg.learning_rate) * \
            0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def get_momentum(step):
        return cfg.ema_start + (cfg.ema_end - cfg.ema_start) * step / max(1, total_steps)

    print(f" Training ({total_steps} steps)...")
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        context_encoder.train()
        predictor.train()
        target_encoder.eval()
        ep_loss, n_bat = 0.0, 0

        for images, labels in train_loader:
            images = images.to(cfg.device)
            B = images.size(0)

            ctx_masks, tgt_masks = patchify(
                B, cfg.num_patches, cfg.num_blocks,
                trg_ratio=tuple(cfg.trg_ratio), ctx_ratio=tuple(cfg.ctx_ratio),
                device=cfg.device)

            ctx_embeds = context_encoder(images, ctx_masks)   # use_corruption=0 -> clean

            with torch.no_grad():
                tgt_full = target_encoder(images)
                tgt_embeds = apply_masks(tgt_full, tgt_masks)
                tgt_embeds = repeat_interleave_batch(tgt_embeds, B, repeat=len(ctx_masks))

            pred_embeds = predictor(ctx_embeds, ctx_masks, tgt_masks)
            loss = F.smooth_l1_loss(pred_embeds, tgt_embeds)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            momentum = get_momentum(global_step)
            update_ema(context_encoder, target_encoder, momentum)

            global_step += 1
            ep_loss += loss.item()
            n_bat += 1

        if epoch % 10 == 0 or epoch == cfg.epochs or epoch == 1:
            print(f"  ep {epoch:03d}/{cfg.epochs}  loss={ep_loss/max(n_bat,1):.4f}")

    context_encoder.eval()
    print(" Training complete.\n")
    return context_encoder


# ═══════════════════════════════════════════════════════════════
#  Step 2 (Option B): genuine/impostor cosine-similarity histograms
# ═══════════════════════════════════════════════════════════════

def compute_genuine_impostor(feature_extractor, gal_samples, prb_samples,
                              id_map, cfg):
    """Mirrors evaluate.py's evaluate_rank1_eer internals, but RETURNS the
    raw genuine/impostor cosine-similarity arrays instead of collapsing
    them to a scalar EER -- evaluate.py itself is not modified."""
    gal_ds = CASIADataset(gal_samples, id_map, cfg.img_size, augment=False)
    prb_ds = CASIADataset(prb_samples, id_map, cfg.img_size, augment=False)
    gal_loader = DataLoader(gal_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    prb_loader = DataLoader(prb_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    gal_feats, gal_labels = extract_features(feature_extractor, gal_loader, cfg.device)
    prb_feats, prb_labels = extract_features(feature_extractor, prb_loader, cfg.device)

    sim = prb_feats @ gal_feats.T
    genuine, impostor = [], []
    glabs = gal_labels.numpy()
    for i in range(len(prb_labels)):
        pid = prb_labels[i].item()
        sims = sim[i].numpy()
        gen_mask = glabs == pid
        imp_mask = glabs != pid
        if gen_mask.any():
            genuine.extend(sims[gen_mask].tolist())
        if imp_mask.any():
            impostor.extend(sims[imp_mask].tolist())
    return np.array(genuine), np.array(impostor)


def build_option_b(cfg, context_encoder, train_loader, eval_dict, id_map):
    print(" ── Option B: genuine/impostor distributions ──")
    feature_extractor = FeatureExtractor(context_encoder)

    modes = {}

    # Mode 1: seen_dom_seen_id -- the TRAINING samples themselves, no
    # resplitting, reusing split_gallery_probe exactly as build_datasets
    # does for the other 3 modes.
    train_samples = train_loader.dataset.samples
    train_id_map = build_id_map(train_samples)   # local, contiguous over training IDs
    gal, prb = split_gallery_probe(train_samples, train_id_map, cfg.gallery_ratio, cfg.seed)
    modes["seen_dom_seen_id"] = compute_genuine_impostor(
        feature_extractor, gal, prb, train_id_map, cfg)

    # Modes 2-4: reuse build_datasets()'s own eval_dict split logic by
    # re-deriving gallery/probe SAMPLE LISTS the same way build_datasets
    # did internally -- eval_dict only exposes loaders, not raw samples,
    # so we rebuild from the same eval_sets via the same split functions
    # already used above (split_gallery_probe), keyed off eval_dict's own
    # loaders is not possible (loaders don't expose .dataset.samples in a
    # gallery/probe-separated form cleanly here), so instead we recompute
    # genuine/impostor directly from eval_dict's existing loaders:
    for name in ("seen_dom_unseen_id", "unseen_dom_seen_id", "unseen_dom_unseen_id"):
        if name not in eval_dict:
            print(f"   (skipping {name}: not present in eval_dict for this config)")
            continue
        ev = eval_dict[name]
        gal_feats, gal_labels = extract_features(feature_extractor, ev["gallery_loader"], cfg.device)
        prb_feats, prb_labels = extract_features(feature_extractor, ev["probe_loader"], cfg.device)
        sim = prb_feats @ gal_feats.T
        genuine, impostor = [], []
        glabs = gal_labels.numpy()
        for i in range(len(prb_labels)):
            pid = prb_labels[i].item()
            sims = sim[i].numpy()
            gen_mask = glabs == pid
            imp_mask = glabs != pid
            if gen_mask.any():
                genuine.extend(sims[gen_mask].tolist())
            if imp_mask.any():
                impostor.extend(sims[imp_mask].tolist())
        modes[name] = (np.array(genuine), np.array(impostor))

    # ─── Plot: 2x2 grid ───
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    titles = {
        "seen_dom_seen_id": "Seen-Domain / Seen-ID\n(training samples)",
        "seen_dom_unseen_id": "Seen-Domain / Unseen-ID",
        "unseen_dom_seen_id": "Unseen-Domain / Seen-ID",
        "unseen_dom_unseen_id": "Unseen-Domain / Unseen-ID",
    }
    for ax, name in zip(axes.flat, titles.keys()):
        if name not in modes:
            ax.set_title(f"{titles[name]}\n(not available)")
            ax.axis("off")
            continue
        genuine, impostor = modes[name]
        ax.hist(genuine, bins=40, alpha=0.6, density=True, label="Genuine", color="tab:green")
        ax.hist(impostor, bins=40, alpha=0.6, density=True, label="Impostor", color="tab:red")
        ax.set_title(titles[name])
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(f"Genuine vs. Impostor Similarity — trained on {cfg.data_dir} "
                 f"({','.join(TRAIN_SPECTRUMS)})", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "option_b_genuine_impostor.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════
#  Step 3 (Option C): t-SNE/UMAP colored by domain
# ═══════════════════════════════════════════════════════════════

def stratified_sample(samples, max_points, key_fn):
    """Two-level stratification: even quota across key_fn(sample) groups
    first, then within each group's quota, maximize identity diversity
    (>=1 image/identity before any identity gets a 2nd)."""
    by_group = defaultdict(list)
    for s in samples:
        by_group[key_fn(s)].append(s)

    groups = list(by_group.keys())
    n_groups = len(groups)
    per_group_quota = max(1, max_points // max(1, n_groups))

    rng = random.Random(SEED)
    picked = []
    for g in groups:
        group_samples = by_group[g]
        by_id = defaultdict(list)
        for s in group_samples:
            by_id[s["identity"]].append(s)
        ids = list(by_id.keys())
        rng.shuffle(ids)

        group_picked = []
        # first pass: 1 per identity
        for ident in ids:
            if len(group_picked) >= per_group_quota:
                break
            group_picked.append(rng.choice(by_id[ident]))
        # second pass: fill remaining quota with extra samples per identity
        idx = 0
        while len(group_picked) < per_group_quota and idx < len(ids) * 5:
            ident = ids[idx % len(ids)]
            pool = by_id[ident]
            if len(pool) > 1:
                group_picked.append(rng.choice(pool))
            idx += 1
        picked.extend(group_picked)

    rng.shuffle(picked)
    return picked[:max_points]


def build_option_c(cfg, context_encoder):
    print(" ── Option C: t-SNE/UMAP by domain ──")

    own_key = normalize_dataset_key(cfg.data_dir)
    dir_by_key = {"casiams": CASIA_DIR, "xjtu": XJTU_DIR, "xpalm": XPALM_DIR}

    all_samples = []
    for key, ddir in dir_by_key.items():
        if not ddir:
            continue
        print(f"  Scanning '{key}' at {ddir} ...")
        samples = scan_by_key(key, ddir)
        for s in samples:
            s = dict(s)          # don't mutate the original dict
            s["dataset"] = key
            all_samples.append(s)

    if not all_samples:
        print("  No samples found across CASIA_DIR/XJTU_DIR/XPALM_DIR -- skipping Option C.")
        return

    key_fn = (lambda s: s["dataset"]) if COLOR_BY == "dataset" else (lambda s: (s["dataset"], s["spectrum"]))
    picked = stratified_sample(all_samples, MAX_TSNE_POINTS, key_fn)
    print(f"  Selected {len(picked)} points (cap={MAX_TSNE_POINTS}) across "
          f"{len(set(key_fn(s) for s in picked))} groups.")

    id_map = build_id_map(picked)
    ds = CASIADataset(picked, id_map, cfg.img_size, augment=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    feature_extractor = FeatureExtractor(context_encoder)
    feats, _ = extract_features(feature_extractor, loader, cfg.device)
    feats = feats.numpy()

    color_labels = [s["dataset"] if COLOR_BY == "dataset" else s["spectrum"] for s in picked]

    print(f"  Running {REDUCTION_METHOD.upper()} on {feats.shape[0]} points "
          f"({feats.shape[1]}-d) ...")
    if REDUCTION_METHOD == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=SEED)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=SEED, init="pca")
    coords = reducer.fit_transform(feats)

    unique_labels = sorted(set(color_labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 8))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(color_labels) if l == lab]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.7,
                  color=label_to_color[lab], label=lab)
    ax.set_title(f"{REDUCTION_METHOD.upper()} of encoder features, colored by "
                f"{COLOR_BY} (trained on {cfg.data_dir}, {','.join(TRAIN_SPECTRUMS)})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, markerscale=2)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"option_c_{REDUCTION_METHOD}_{COLOR_BY}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = build_cfg()
    set_seed(cfg.seed)

    print(f"\n{'='*70}\n DOMAIN DISTRIBUTION ANALYSIS\n{'='*70}")
    print(f" data_dir={cfg.data_dir}  train_spectrums={TRAIN_SPECTRUMS}  "
          f"test_spectrums={TEST_SPECTRUMS}\n{'='*70}\n")

    train_loader, eval_dict, id_map, n_train_ids, train_id_map = build_datasets(cfg)

    context_encoder = train_plain_jepa(cfg, train_loader)

    build_option_b(cfg, context_encoder, train_loader, eval_dict, id_map)
    build_option_c(cfg, context_encoder)

    print(f"\n{'='*70}\n DONE. Outputs in {OUTPUT_DIR}\n{'='*70}")


if __name__ == "__main__":
    main()
