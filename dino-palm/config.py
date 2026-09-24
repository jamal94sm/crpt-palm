"""config.py -- DINO on CASIA-MS palmprints.

All values below verified against facebookresearch/dino/main_dino.py
(fetched directly): momentum_teacher=0.996->1.0 cosine, out_dim=65536
(scaled down here, see models.py), warmup_teacher_temp=0.04,
teacher_temp=0.04, warmup_teacher_temp_epochs=30 (scaled to --epochs
ratio here), lr=0.0005 (base, linear-scaling-rule at ref batch 256),
warmup_epochs=10 (scaled to ratio), weight_decay=0.04->0.4 cosine,
clip_grad=3.0, freeze_last_layer=1 epoch, student_temp=0.1,
center_momentum=0.9. global_crops_scale=(0.4,1.0), local_crops_scale=
(0.05,0.4), local_crops_number=8 -- all official defaults, unchanged.
Crop PIXEL SIZES are scaled proportionally to this project's img_size
(see multicrop_dataset.py) since official's 224/96px are ImageNet-scale.
"""

import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="DINO on CASIA-MS")

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
    p.add_argument("--dino_out_dim", type=int, default=None,
        help="Official uses 65536 (ImageNet-scale). Defaults to "
             "embed_dim*8 here, proportionally scaled down.")
    p.add_argument("--use_bn_in_head", type=int, default=0, choices=[0, 1])
    p.add_argument("--norm_last_layer", type=int, default=1, choices=[0, 1])

    # ─── Multi-crop (official defaults) ─────────────────────────
    p.add_argument("--global_crops_scale", type=float, nargs=2, default=[0.4, 1.0])
    p.add_argument("--local_crops_scale", type=float, nargs=2, default=[0.05, 0.4])
    p.add_argument("--local_crops_number", type=int, default=8)

    # ─── DINO loss (official defaults) ──────────────────────────
    p.add_argument("--momentum_teacher", type=float, default=0.996)
    p.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--warmup_teacher_temp_epochs_ratio", type=float, default=0.15,
        help="Official: 30/~100-300 epochs -- scaled to --epochs here.")
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--center_momentum", type=float, default=0.9)
    p.add_argument("--freeze_last_layer_epochs", type=int, default=1)

    # ─── Optimizer (official defaults) ──────────────────────────
    p.add_argument("--base_lr", type=float, default=0.0005)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs_ratio", type=float, default=0.1,
        help="Official: 10 epochs -- scaled to --epochs ratio here.")
    p.add_argument("--dino_wd_start", type=float, default=0.04)
    p.add_argument("--dino_wd_end", type=float, default=0.4)
    p.add_argument("--clip_grad", type=float, default=3.0)

    # ─── Training ─────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.05)   # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--learning_rate", type=float, default=1e-3)  # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--warmup_ratio", type=float, default=0.1)    # SHARED_ARG_NAMES compat, UNUSED

    # ─── Evaluation ───────────────────────────────────────────
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_use_cls", type=int, default=1, choices=[0, 1],
        help="1 (default, official convention) = eval via CLS token. "
             "0 = mean-pool patch tokens instead, for consistency with "
             "this project's other baselines.")

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_dino")

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
