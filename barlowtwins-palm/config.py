"""
config.py — Barlow Twins on CASIA-MS palmprints.

Verified against facebookresearch/barlowtwins/main.py directly (fetched
2026-09-03). No predictor, no EMA/target network -- a single shared
encoder+projector, fully symmetric, no stop-gradient anywhere.
"""

import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="Barlow Twins on CASIA-MS")

    # ─── Dataset ──────────────────────────────────────────────
    p.add_argument("--data_dir", required=True,
        default="/home/pai-ng/Jamal/CASIA-MS-ROI")
    p.add_argument("--img_size", type=int, default=112)

    # ─── Mode ─────────────────────────────────────────────────
    p.add_argument("--mode", default="all",
        choices=["all", "cross_domain", "cross_domain_openset"])
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"])
    p.add_argument("--train_id_ratio", type=float, default=0.8)
    p.add_argument("--test_sample_ratio", type=float, default=0.2)
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--aug_multiplier", type=int, default=8)

    # ─── Encoder architecture ──────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8)

    # ─── Barlow Twins projector + loss (official values verified) ──
    p.add_argument("--projector_hidden_dim", type=int, default=None,
        help="Official value is a FIXED 8192 (sized for a 2048-dim "
             "ResNet-50, ratio 4x) -- inappropriate for this project's "
             "much smaller ViT embed_dim. Default (None) resolves to "
             "embed_dim.")
    p.add_argument("--projector_out_dim", type=int, default=None,
        help="Official Barlow Twins keeps ALL THREE projector layers the "
             "SAME width (8192-8192-8192) -- unlike BYOL, hidden and "
             "output are equal here. Default (None) resolves to embed_dim "
             "(same value as --projector_hidden_dim's default), matching "
             "that equal-width convention. WARNING: the cross-correlation "
             "loss term is a (proj_dim x proj_dim) matrix estimated from "
             "the batch -- if --batch_size is much smaller than the "
             "resolved proj_dim, the matrix is under-determined (the "
             "exact conditioning issue diagnosed for VICReg's covariance "
             "term earlier in this project). Keep --batch_size >= "
             "proj_dim, or lower --projector_out_dim, to avoid it "
             "from the start.")
    p.add_argument("--lambd", type=float, default=0.0051,
        help="Official weight on the off-diagonal (redundancy-reduction) "
             "term.")

    # ─── LARS optimizer (official, verified) ────────────────────
    p.add_argument("--learning_rate_weights", type=float, default=0.2,
        help="Official base LR multiplier for weight parameters "
             "(ndim > 1).")
    p.add_argument("--learning_rate_biases", type=float, default=0.0048,
        help="Official base LR multiplier for biases/BN parameters "
             "(ndim == 1) -- note this is a SEPARATE, much smaller LR "
             "than weights, a real structural difference from VICReg/"
             "BYOL's single shared LR.")
    p.add_argument("--lars_wd", type=float, default=1e-6,
        help="Official weight decay for LARS.")
    p.add_argument("--lars_momentum", type=float, default=0.9)
    p.add_argument("--lars_eta", type=float, default=0.001,
        help="LARS trust coefficient (official default, hardcoded in "
             "their LARS class rather than exposed as a flag -- exposed "
             "here for consistency with this project's other baselines).")

    # ─── Training ─────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--start_lr", type=float, default=1e-5)
    p.add_argument("--final_lr", type=float, default=1e-6)
    p.add_argument("--final_weight_decay", type=float, default=0.4)

    # ─── Evaluation ───────────────────────────────────────────
    p.add_argument("--eval_every", type=int, default=10)

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_barlowtwins")

    p.add_argument("--use_CI", type=int, default=0, choices=[0, 1])
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--use_cross_dataset_eval", type=int, default=0, choices=[0, 1])
    p.add_argument("--casia_dir", type=str, default=None)
    p.add_argument("--xjtu_dir", type=str, default=None)
    p.add_argument("--xpalm_dir", type=str, default=None)
    p.add_argument("--test_spectrums", nargs="*", default=None)

    cfg = p.parse_args(args)
    return cfg
