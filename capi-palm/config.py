"""config.py -- CAPI on palmprints (Darcet et al., TMLR 2025, arXiv:2502.08769).
Verified directly against facebookresearch/capi's model.py and train_capi.py
(both read in full). Hyperparameters follow the paper's Table 7 where scale-
independent; scale-dependent values (batch size, prototype count) are
explicitly ADAPTED for this project's much smaller batch/dataset.
"""
import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="CAPI on palmprints")

    p.add_argument("--data_dir", required=True, default="/home/pai-ng/Jamal/CASIA-MS-ROI")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--mode", default="all",
        choices=["all", "cross_domain", "cross_domain_openset", "cross_brand_openset"])
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"])
    p.add_argument("--test_spectrums", nargs="*", default=None)
    p.add_argument("--train_brands", nargs="*", default=["iPhone"])
    p.add_argument("--test_brands", nargs="*", default=None)
    p.add_argument("--train_id_ratio", type=float, default=0.8)
    p.add_argument("--test_sample_ratio", type=float, default=0.2)
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--aug_multiplier", type=int, default=8)

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8)
    p.add_argument("--vit_depth", type=int, default=None)
    p.add_argument("--vit_heads", type=int, default=None)
    p.add_argument("--n_registers", type=int, default=4,
        help="ADAPTED from paper's 16 (must be divisible by 4 for edge "
             "placement); scaled down for this project's small grid.")
    p.add_argument("--drop_path_rate", type=float, default=0.1,
        help="ADAPTED. Paper: 0.2 (Table 7), for a 300M ViT-L; lowered "
             "for this project's much smaller encoder.")

    p.add_argument("--mask_ratio", type=float, default=0.65, help="Paper's own value (Table 1e ablation optimum).")
    p.add_argument("--prediction_subsampling", type=float, default=0.055,
        help="Fraction of MASKED tokens to predict per image (source's "
             "data.py: n_predict = int(n_masked * prediction_subsampling), "
             "NOT a fixed count -- an earlier version of this baseline "
             "wrongly hardcoded n_predict=7 from the paper's ViT-L/14@224 "
             "example. 0.055 reproduces ~7 predictions at the paper's own "
             "scale (41 masked patches on a 14x14 grid at 65%% mask "
             "ratio); at this project's smaller grids the ABSOLUTE number "
             "of predicted patches will be smaller too unless raised.")
    p.add_argument("--mask_roll", type=int, default=1, choices=[0, 1], help="Paper's own '+roll' fix (Section 4.2).")
    p.add_argument("--crop_scale", type=float, nargs=2, default=[0.6, 1.0], help="Paper's own optimum (Table 1d).")

    p.add_argument("--num_prototypes", type=int, default=2048,
        help="ADAPTED. Paper: 16384 (Table 7), tuned for ViT-L/inet1k. "
             "Scaled down to match this project's DINOv2-palm baseline's "
             "own 8*embed_dim convention for prototype count at this scale.")
    p.add_argument("--n_sk_iter", type=int, default=3, help="Paper's own value.")
    p.add_argument("--target_temp", type=float, default=0.06, help="Paper's own value (Table 7).")
    p.add_argument("--pred_temp", type=float, default=0.12, help="Paper's own value (Table 7, 'student temperature').")
    p.add_argument("--positionwise_sk", type=int, default=1, choices=[0, 1],
        help="Paper's own fix for positional collapse (Section 3.1). Leave ON.")

    p.add_argument("--base_lr", type=float, default=1e-3, help="Paper's own value (Table 7).")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.1, help="Paper's own value (Table 7).")
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.95, help="Paper's own value (Table 7).")
    p.add_argument("--warmup_epochs_ratio", type=float, default=0.1, help="Paper's own value ('10%').")
    p.add_argument("--cosine_truncation", type=float, default=0.2, help="Paper's own value: truncate last 20% of cosine.")
    p.add_argument("--teacher_momentum", type=float, default=None,
        help="None = paper's own rule mu = 1 - lr (computed per-step from the LR schedule). "
             "Set explicitly to override with a fixed value instead.")
    p.add_argument("--clustering_lr_mult", type=float, default=0.5, help="Paper's own value ('half of the backbone lr').")
    p.add_argument("--patch_embed_lr_mult", type=float, default=0.2, help="Paper's own value (Table 7).")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)  # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--warmup_ratio", type=float, default=0.1)    # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--eval_every", type=int, default=10)

    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_capi")
    p.add_argument("--use_CI", type=int, default=0, choices=[0, 1])
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--use_cross_dataset_eval", type=int, default=0, choices=[0, 1])
    p.add_argument("--casia_dir", type=str, default=None)
    p.add_argument("--xjtu_dir", type=str, default=None)
    p.add_argument("--xpalm_dir", type=str, default=None)

    return p.parse_args(args)
