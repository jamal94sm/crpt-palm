"""config.py -- DINOv2-style self-supervised pretraining on palmprints.

Objective + hyper-parameters follow facebookresearch/dinov2 (files read directly:
ssl_default_config.yaml, train/ssl_meta_arch.py, train/train.py, loss/ibot_patch_loss.py,
loss/koleo_loss.py, data/collate.py, data/masking.py __init__/__call__, and the
prepare_tokens_with_masks() excerpt of models/vision_transformer.py). Values that differ
from the official ones are deliberate adaptations to this project (tiny ViT, ~1.6k training
images, batch 64, 112px) and are marked ADAPTED in the help strings.
"""
import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="DINOv2-style pretraining on palmprints")

    # ─── Dataset / split (identical to the other sibling baselines) ────
    p.add_argument("--data_dir", required=True, default="/home/pai-ng/Jamal/CASIA-MS-ROI")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--mode", default="all",
        choices=["all", "cross_domain", "cross_domain_openset", "cross_brand_openset", "cross_dataset"])
    # ─── cross_dataset mode ───────────────────────────────────
    p.add_argument("--train_datasets", nargs="*", default=None,
                   choices=["casiams", "xjtu", "xpalm"],
                   help="cross_dataset mode: datasets used for training "
                        "(all subsets, all identities, concatenated).")
    p.add_argument("--test_datasets", nargs="*", default=None,
                   choices=["casiams", "xjtu", "xpalm"],
                   help="cross_dataset mode: unseen datasets evaluated at the end "
                        "of training. Default: every dataset not in --train_datasets.")
    p.add_argument("--xpalm_scanner", type=int, default=1, choices=[0, 1],
                   help="cross_dataset mode: 1 = include the X-Palm scanner images, "
                        "0 = smartphone images only (applies to X-Palm as train and as test).")
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"])
    p.add_argument("--test_spectrums", nargs="*", default=None)
    p.add_argument("--train_brands", nargs="*", default=["iPhone"])
    p.add_argument("--test_brands", nargs="*", default=None)
    p.add_argument("--train_id_ratio", type=float, default=0.8)
    p.add_argument("--test_sample_ratio", type=float, default=0.2)
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--aug_multiplier", type=int, default=8)

    # ─── Backbone (capacity-matched to ContextEncoder / PlainViT) ──────
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8)
    p.add_argument("--vit_depth", type=int, default=None,
        help="None = same auto rule as ContextEncoder: min(6, embed_dim//64 + 2).")
    p.add_argument("--vit_heads", type=int, default=None,
        help="None = largest divisor of embed_dim not above embed_dim//32.")
    p.add_argument("--drop_path_rate", type=float, default=0.1,
        help="ADAPTED. Official default 0.3 is for a 300M ViT-L; uniform across blocks "
             "(official drop_path_uniform=true). 0.1 = DINO v1 default for small ViTs.")
    p.add_argument("--layerscale_init", type=float, default=1e-5,
        help="Official 1e-5. <=0 disables LayerScale.")

    # ─── Projection head ────────────────────────────────────────────────
    p.add_argument("--dino_out_dim", type=int, default=None,
        help="ADAPTED. Official 65536 prototypes (ImageNet-scale). None = 8*embed_dim.")
    p.add_argument("--head_nlayers", type=int, default=3, help="Official 3.")
    p.add_argument("--head_hidden_dim", type=int, default=None,
        help="ADAPTED. Official 2048 for ViT-L (2x embed). None = 2*embed_dim.")
    p.add_argument("--head_bottleneck_dim", type=int, default=None,
        help="ADAPTED. Official 256 for ViT-L (1/4 embed). None = max(embed_dim//4, 32).")
    p.add_argument("--ibot_separate_head", type=int, default=0, choices=[0, 1],
        help="Official default 0 (iBOT patch tokens share the DINO head).")

    # ─── Multi-crop (official scales; sizes derived from the patch grid) ──
    p.add_argument("--global_crops_scale", type=float, nargs=2, default=[0.32, 1.0], help="Official.")
    p.add_argument("--local_crops_scale", type=float, nargs=2, default=[0.05, 0.32], help="Official.")
    p.add_argument("--local_crops_number", type=int, default=8, help="Official 8.")

    # ─── Losses ─────────────────────────────────────────────────────────
    p.add_argument("--dino_loss_weight", type=float, default=1.0)
    p.add_argument("--ibot_loss_weight", type=float, default=1.0)
    p.add_argument("--koleo_loss_weight", type=float, default=0.1, help="Official 0.1.")
    p.add_argument("--ibot_mask_ratio_min", type=float, default=0.1, help="Official.")
    p.add_argument("--ibot_mask_ratio_max", type=float, default=0.5, help="Official.")
    p.add_argument("--ibot_mask_sample_probability", type=float, default=0.5, help="Official.")
    p.add_argument("--student_temp", type=float, default=0.1, help="Official.")
    p.add_argument("--center_momentum", type=float, default=0.9, help="Official.")

    # ─── Teacher ────────────────────────────────────────────────────────
    p.add_argument("--momentum_teacher", type=float, default=0.996,
        help="ADAPTED. Official 0.992 was tuned for batches of >=2k images; DINO v1 advises "
             "a HIGHER momentum for smaller batches (ours: 64), so 0.996 (same as dino-palm).")
    p.add_argument("--final_momentum_teacher", type=float, default=1.0, help="Official.")
    p.add_argument("--warmup_teacher_temp", type=float, default=0.04, help="Official.")
    p.add_argument("--teacher_temp", type=float, default=0.07, help="Official.")
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=30,
        help="Official 30; capped at 30%% of --epochs so short runs stay valid.")

    # ─── Optimisation (official values; LR uses the official sqrt-wrt-1024 rule) ──
    p.add_argument("--base_lr", type=float, default=0.004,
        help="Official: lr = base_lr * sqrt(batch_size / 1024)  (=1e-3 at batch 64).")
    p.add_argument("--min_lr", type=float, default=1e-6, help="Official.")
    p.add_argument("--warmup_epochs_ratio", type=float, default=0.1,
        help="Official 10 of 100 epochs, expressed as a ratio.")
    p.add_argument("--wd_start", type=float, default=0.04, help="Official.")
    p.add_argument("--wd_end", type=float, default=0.4, help="Official.")
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.999)
    p.add_argument("--clip_grad", type=float, default=3.0, help="Official, per module.")
    p.add_argument("--freeze_last_layer_epochs", type=int, default=1, help="Official.")
    p.add_argument("--layerwise_decay", type=float, default=0.9, help="Official.")
    p.add_argument("--patch_embed_lr_mult", type=float, default=0.2, help="Official.")

    # ─── Training / eval ────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.05)   # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--learning_rate", type=float, default=1e-3)  # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--warmup_ratio", type=float, default=0.1)    # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_use_cls", type=int, default=1, choices=[0, 1],
        help="1 = evaluate the TEACHER CLS token (official). 0 = mean-pool teacher patch tokens.")

    # ─── Misc / ecosystem plumbing ──────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_dinov2")
    p.add_argument("--use_CI", type=int, default=0, choices=[0, 1])
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--use_cross_dataset_eval", type=int, default=0, choices=[0, 1])
    p.add_argument("--casia_dir", type=str, default=None)
    p.add_argument("--xjtu_dir", type=str, default=None)
    p.add_argument("--xpalm_dir", type=str, default=None)

    return p.parse_args(args)
