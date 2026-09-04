"""
config.py — BYOL on CASIA-MS palmprints.

Mirrors crpt-palm's config.py section layout so hyperparameters line up 1:1
for a fair comparison. JEPA/corruption/Gabor/struct/supervision flags are
dropped (BYOL doesn't use them). Unlike VICReg, BYOL DOES have a target
network (EMA of the online encoder+projector) -- see --ema_start/--ema_end.
"""

import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="BYOL on CASIA-MS")

    # ─── Dataset ──────────────────────────────────────────────
    p.add_argument("--data_dir", required=True,
        default="/home/pai-ng/Jamal/CASIA-MS-ROI")
    p.add_argument("--img_size", type=int, default=112)

    # ─── Mode ─────────────────────────────────────────────────
    p.add_argument("--mode", default="all",
        choices=["all", "cross_domain", "cross_domain_openset"],
        help="'all' = all domains+IDs, "
             "'cross_domain' = selected domains, all IDs, "
             "'cross_domain_openset' = selected domains+IDs")
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"],
        help="Spectrums for training (cross_domain modes)")
    p.add_argument("--train_id_ratio", type=float, default=0.8,
        help="Fraction of IDs for training (openset mode)")
    p.add_argument("--test_sample_ratio", type=float, default=0.2,
        help="Fraction of training samples held out for eval")
    p.add_argument("--gallery_ratio", type=float, default=0.5,
        help="Fraction of test samples used as gallery")
    p.add_argument("--aug_multiplier", type=int, default=8,
        help="Augmentation multiplier for training data -- the SAME "
             "transform pipeline BYOL's paired views are drawn from "
             "(see paired_dataset.py). Must match the proposed method's "
             "--aug_multiplier for a fair comparison.")

    # ─── Encoder architecture ──────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8,
        help="Grid size (8 → 8×8=64 patches for 112px). Encoder depth/heads "
             "are auto-derived from embed_dim, same formula as the proposed "
             "method's context/target encoders, so capacity matches 1:1.")

    # ─── BYOL projector + predictor ─────────────────────────────
    p.add_argument("--projector_hidden_dim", type=int, default=None,
        help="BYOL projector hidden dim. Official paper value is a FIXED "
             "4096 (sized for a 2048-dim ResNet-50 trunk) -- inappropriate "
             "for this project's much smaller ViT embed_dim. Default "
             "(None) resolves to embed_dim, matching the ~1x hidden/"
             "encoder-dim ratio used by smaller-backbone BYOL ports "
             "(sthalles/PyTorch-BYOL, YoadTew/BYOL-Autoencoder: hidden=512 "
             "for a 512-dim ResNet-18).")
    p.add_argument("--projector_out_dim", type=int, default=None,
        help="BYOL projection/prediction output size. Official paper value "
             "is a fixed 256. Default (None) resolves to "
             "max(embed_dim // 4, 32), matching the ~0.25x projection/"
             "encoder-dim ratio used by smaller-backbone BYOL ports "
             "(projection=128 for a 512-dim ResNet-18).")
    p.add_argument("--predictor_hidden_dim", type=int, default=None,
        help="BYOL predictor hidden dim. Same reasoning/default as "
             "--projector_hidden_dim -- resolves to embed_dim.")
    p.add_argument("--ema_start", type=float, default=0.996,
        help="BYOL target-network momentum at step 0 (paper default).")
    p.add_argument("--ema_end", type=float, default=1.0,
        help="BYOL target-network momentum at the final step (paper uses "
             "a cosine schedule up to 1.0).")

    # ─── LARS optimizer (official BYOL, paper Appendix J) ───────
    p.add_argument('--base_lr', type=float, default=0.2,
        help="Official BYOL base LR. Effective LR after a 10-epoch "
             "warmup = base_lr * batch_size / 256, then cosine decay.")
    p.add_argument('--lars_wd', type=float, default=1.5e-6,
        help="Official BYOL weight decay for LARS (paper Appendix J -- "
             "note this differs from VICReg's 1e-6).")
    p.add_argument('--lars_momentum', type=float, default=0.9)
    p.add_argument('--lars_eta', type=float, default=0.001,
        help="LARS trust coefficient (paper Appendix J: trust_coef=1e-3).")

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
    p.add_argument("--output_dir", default="./output_byol")

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
