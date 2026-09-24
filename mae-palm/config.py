"""config.py -- MAE on CASIA-MS palmprints.

Optimizer/schedule verified against facebookresearch/mae's PRETRAIN.md
(fetched directly): AdamW, weight_decay=0.05, base_lr(blr)=1.5e-4 (linear
scaling rule: lr = blr * batch_size/256), warmup_epochs=40/800=5% (scaled
to --epochs here), mask_ratio=0.75, norm_pix_loss=True (official's own
recommended setting, not the default-off pixel target).
NOTE: AdamW betas=(0.9, 0.95) is the standard MAE/ViT-pretraining
convention widely used across facebookresearch SSL repos, but was NOT
independently re-confirmed via a direct fetch of the exact beta values in
this session -- flagged here rather than silently asserted with the same
confidence as the fetched values above.
"""

import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="MAE on CASIA-MS")

    # ─── Dataset ──────────────────────────────────────────────
    p.add_argument("--data_dir", required=True, default="/home/pai-ng/Jamal/CASIA-MS-ROI")
    p.add_argument("--img_size", type=int, default=112)

    # ─── Mode ─────────────────────────────────────────────────
    p.add_argument("--mode", default="all",
        choices=["all", "cross_domain", "cross_domain_openset",
                 "cross_brand_openset"])
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"])
    p.add_argument("--test_spectrums", nargs="*", default=None)
    p.add_argument("--train_brands", nargs="*", default=["iPhone"])
    p.add_argument("--test_brands", nargs="*", default=None)
    p.add_argument("--train_id_ratio", type=float, default=0.8)
    p.add_argument("--test_sample_ratio", type=float, default=0.2)
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--aug_multiplier", type=int, default=8)

    # ─── Architecture ─────────────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8)
    p.add_argument("--decoder_dim", type=int, default=None,
        help="Defaults to ~2/3 of embed_dim, matching official's "
             "512/768 encoder/decoder ratio.")
    p.add_argument("--decoder_depth", type=int, default=4,
        help="Official uses 8 at ImageNet scale (encoder depth 12); "
             "scaled down for this project's shallower 6-depth encoder.")
    p.add_argument("--decoder_heads", type=int, default=None)

    # ─── MAE-specific ─────────────────────────────────────────
    p.add_argument("--mask_ratio", type=float, default=0.75,
        help="Official default (paper: 75%% for images).")
    p.add_argument("--norm_pix_loss", type=int, default=1, choices=[0, 1],
        help="Official's own recommended setting (PRETRAIN.md: 'we use "
             "--norm_pix_loss as the target for better representation "
             "learning').")

    # ─── Optimizer (verified: AdamW, official PRETRAIN.md values) ──────
    p.add_argument("--base_lr", type=float, default=1.5e-4,
        help="Official blr (PRETRAIN.md): lr = blr * batch_size / 256.")
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.95,
        help="Standard MAE/ViT-pretraining convention -- NOT independently "
             "re-verified via direct fetch in this session, unlike base_lr/"
             "weight_decay/warmup_epochs/mask_ratio above.")
    p.add_argument("--adamw_wd", type=float, default=0.05)
    p.add_argument("--warmup_epochs_ratio", type=float, default=0.05,
        help="Official: 40/800 epochs = 0.05, scaled to --epochs here.")

    # ─── Training ─────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.05)   # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--learning_rate", type=float, default=1e-3)  # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--warmup_ratio", type=float, default=0.1)    # SHARED_ARG_NAMES compat, UNUSED

    # ─── Evaluation ───────────────────────────────────────────
    p.add_argument("--eval_every", type=int, default=10)

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_mae")

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
