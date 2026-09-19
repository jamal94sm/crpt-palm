"""config.py -- MaskFeat on CASIA-MS palmprints.

Optimizer/schedule verified against open-mmlab/mmpretrain's
maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k.py (fetched directly,
2026-09-18): AdamW, betas=(0.9,0.999), wd=0.05, no-decay on bias/norm/
mask_token, grad-clip max_norm=0.02, linear warmup -> cosine, mask
ratio=0.4 (block-wise). HOG: nbins=9, pool adapted to 7 (see hog.py) for
this project's 14px patch size (image_size=112, num_patches=8), gaussian_
window adapted to 14 accordingly.
"""

import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="MaskFeat on CASIA-MS")

    # ─── Dataset ──────────────────────────────────────────────
    p.add_argument("--data_dir", required=True, default="/home/pai-ng/Jamal/CASIA-MS-ROI")
    p.add_argument("--img_size", type=int, default=112)

    # ─── Mode ─────────────────────────────────────────────────
    p.add_argument("--mode", default="all",
        choices=["all", "cross_domain", "cross_domain_openset"])
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"])
    p.add_argument("--test_spectrums", nargs="*", default=None)
    p.add_argument("--train_id_ratio", type=float, default=0.8)
    p.add_argument("--test_sample_ratio", type=float, default=0.2)
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--aug_multiplier", type=int, default=8)

    # ─── Architecture ─────────────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8)

    # ─── MaskFeat-specific ──────────────────────────────────────
    p.add_argument("--mask_ratio", type=float, default=0.4,
        help="Official default (paper Sec 4/Table; mmpretrain config uses "
             "78/196 patches ~= 0.4).")
    p.add_argument("--min_mask_block", type=int, default=4,
        help="Minimum block-mask area (in patches) per sampled block.")
    p.add_argument("--hog_nbins", type=int, default=9)
    p.add_argument("--hog_pool", type=int, default=7,
        help="Adapted from official's 8 (tuned for 16px patches) to 7 "
             "(divides this project's 14px patch size evenly).")
    p.add_argument("--hog_gaussian_window", type=int, default=14,
        help="Adapted from official's 16 to match this project's "
             "image_size/num_patches grid.")

    # ─── Optimizer (verified: AdamW, official values) ────────────
    p.add_argument("--base_lr", type=float, default=2e-4,
        help="Official base LR (mmpretrain config: 2e-4 * 8 at batch "
             "2048 -> 2e-4 at the linear-scaling-rule base of 256).")
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.999)
    p.add_argument("--adamw_wd", type=float, default=0.05)
    p.add_argument("--grad_clip_norm", type=float, default=0.02,
        help="Official value -- unusually aggressive, verified not a typo "
             "(mmpretrain config: clip_grad=dict(max_norm=0.02)).")
    p.add_argument("--warmup_epochs_ratio", type=float, default=0.1,
        help="Official: 30/300 epochs = 0.1 -- scaled to --epochs here "
             "rather than a fixed 30.")

    # ─── Training ─────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.05)   # kept for SHARED_ARG_NAMES compat
    p.add_argument("--learning_rate", type=float, default=1e-3)  # kept for SHARED_ARG_NAMES compat, UNUSED (see base_lr)
    p.add_argument("--warmup_ratio", type=float, default=0.1)    # kept for SHARED_ARG_NAMES compat, UNUSED

    # ─── Evaluation ───────────────────────────────────────────
    p.add_argument("--eval_every", type=int, default=10)

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_maskfeat")

    p.add_argument("--use_CI", type=int, default=0, choices=[0, 1])
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--use_cross_dataset_eval", type=int, default=0, choices=[0, 1])
    p.add_argument("--casia_dir", type=str, default=None)
    p.add_argument("--xjtu_dir", type=str, default=None)
    p.add_argument("--xpalm_dir", type=str, default=None)

    cfg = p.parse_args(args)
    return cfg