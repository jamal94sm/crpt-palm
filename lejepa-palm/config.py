"""config.py -- LeJEPA on palmprints (Balestriero & LeCun, arXiv:2511.08544).

No predictor, no target/EMA encoder, no masking -- see models.py's docstring.
Hyperparameters follow the paper's "Experiment Details 1" and Table 1
ablations (all read directly from the arXiv PDF), with explicit ADAPTED
notes where this project's scale (batch ~64, ~1-2k training images, ViT
~4.91M params) departs from the paper's own validated range (batch >=128,
ImageNet-scale, ViT-L/ViT-H).
"""
import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="LeJEPA on palmprints")

    # ─── Dataset / split (identical to every other sibling baseline) ───
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

    # ─── Encoder (capacity-matched to this project's ContextEncoder) ───
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8)
    p.add_argument("--vit_depth", type=int, default=None,
        help="None = same auto rule as ContextEncoder: min(6, embed_dim//64 + 2).")
    p.add_argument("--vit_heads", type=int, default=None,
        help="None = largest divisor of embed_dim not above embed_dim//32.")

    # ─── Multi-crop (paper's Table 1b: V=8, Vg=2 is the recommended point) ──
    p.add_argument("--global_crops_scale", type=float, nargs=2, default=[0.32, 1.0],
        help="ADAPTED: paper doesn't state exact scale ranges; using the "
             "standard DINO/DINOv2 convention it explicitly follows for "
             "multi-crop setup ('adopt the DINO setup').")
    p.add_argument("--local_crops_scale", type=float, nargs=2, default=[0.05, 0.32])
    p.add_argument("--local_crops_number", type=int, default=6,
        help="Paper Table 1b: V=8 total (2 global + 6 local) is the best "
             "entry actually reported at Vg=2 in that table.")

    # ─── SIGReg (Epps-Pulley) -- paper's own defaults + explicit adaptation ──
    p.add_argument("--lejepa_lambda", type=float, default=0.05,
        help="Paper's single trade-off hyperparameter (Eq. LeJEPA). Paper "
             "recommends 0.05 as 'a robust default' (Section 6.1) after "
             "showing performance is stable across lambda (Figure 8).")
    p.add_argument("--sigreg_num_slices", type=int, default=256,
        help="ADAPTED. Paper recommends 1024 slices for ViT-L/inet1k "
             "(Table 1d) but also shows 512 is competitive and even 16 "
             "works with per-step resampling (Section 4.3, 'SGD beats the "
             "curse of dimensionality'). Lowered further here since this "
             "project's much smaller batch size (~64 vs. paper's >=128) "
             "and embedding dim reduce how many directions are needed to "
             "usefully constrain the space.")
    p.add_argument("--sigreg_integration_bound", type=float, default=5.0,
        help="Paper's own recommended value (Table 1a shows negligible "
             "sensitivity to this choice).")
    p.add_argument("--sigreg_num_points", type=int, default=17,
        help="Paper's own recommended value (Table 1a: negligible effect).")

    # ─── Optimization (paper's "Experiment Details 1") ─────────────────
    p.add_argument("--base_lr", type=float, default=5e-4,
        help="Paper: lr in {5e-3, 5e-4}; 5e-4 chosen as the more "
             "conservative of the two given this project's much smaller "
             "batch size than the paper's validated range.")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2,
        help="Paper: wd in {1e-1, 1e-2, 1e-5}, no scheduler on weight "
             "decay (unlike DINOv2's cosine wd schedule -- LeJEPA "
             "explicitly uses a CONSTANT weight decay, confirmed from "
             "'Experiment Details 1': 'no scheduler on weight-decay').")
    p.add_argument("--warmup_epochs_ratio", type=float, default=0.1,
        help="ADAPTED. Paper says 'standard linear warm-up cosine-"
             "annealing for lr' without stating an exact warmup length; "
             "10%% matches this project's convention for every other "
             "sibling baseline.")
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.999)

    # ─── Training / eval ────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)  # SHARED_ARG_NAMES compat; see --lejepa_batch_size
    p.add_argument("--lejepa_batch_size", type=int, default=256,
        help="LeJEPA's OWN batch size, used in place of --batch_size for "
             "this baseline's DataLoader. Necessary because --run_all_"
             "baselines forwards --batch_size uniformly to every baseline "
             "via SHARED_ARG_NAMES, which silently overrides any default "
             "set on --batch_size itself. LeJEPA's prediction-loss "
             "gradient scales inversely with batch size and collapses at "
             "this project's usual sweep batch size (16-64) -- see "
             "main.py's docstring. Paper's own validated floor is 128; "
             "256 chosen here for extra margin.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)  # SHARED_ARG_NAMES compat, UNUSED (see base_lr)
    p.add_argument("--warmup_ratio", type=float, default=0.1)    # SHARED_ARG_NAMES compat, UNUSED
    p.add_argument("--eval_every", type=int, default=10)

    # ─── Misc / ecosystem plumbing ──────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_lejepa")
    p.add_argument("--use_CI", type=int, default=0, choices=[0, 1])
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--use_cross_dataset_eval", type=int, default=0, choices=[0, 1])
    p.add_argument("--casia_dir", type=str, default=None)
    p.add_argument("--xjtu_dir", type=str, default=None)
    p.add_argument("--xpalm_dir", type=str, default=None)

    return p.parse_args(args)
