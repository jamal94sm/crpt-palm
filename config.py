"""
config.py — JEPA on CASIA-MS: 3 evaluation modes.
"""
import argparse
import json

from gabor import BASE_SCALE_LADDER

def get_cfg(args=None):
    p = argparse.ArgumentParser(description="JEPA on CASIA-MS")

    # ─── Dataset ──────────────────────────────────────────────

    #"data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    #"xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    #"xpalm_data_root"  : "/home/pai-ng/Jamal/xpalm",

    p.add_argument("--data_dir", required=True, default = "/home/pai-ng/Jamal/CASIA-MS-ROI")
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
                   help="Augmentation multiplier for training data")

    # ─── Method toggle ────────────────────────────────────────
    p.add_argument("--method", default="jepa",
                   choices=["jepa", "compnet", "vit_sup"])
    # ─── CompNet (supervised) ─────────────────────────────────
    p.add_argument("--compnet_channels", type=int, default=16,
                   help="base channel width of the Gabor/competition block")
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # ─── JEPA architecture ────────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8,
                   help="Grid size (8 → 8×8=64 patches for 112px)")
    p.add_argument("--num_blocks", type=int, default=2,
                   help="Number of target mask blocks")
    p.add_argument("--trg_ratio", type=float, nargs=2,
                   default=[0.10, 0.15])
    p.add_argument("--ctx_ratio", type=float, nargs=2,
                   default=[0.90, 1.00])

    ### Custom ViT for supervised pretraining
    p.add_argument("--patch_size", type=int, default=14,
                   help="ViT patch size; img_size must be divisible by it "
                        "(112/14 = 8x8 = 64 patches)")
    p.add_argument("--vit_depth", type=int, default=6)
    p.add_argument("--vit_heads", type=int, default=8)

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
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)

    # ─── Evaluation ───────────────────────────────────────────
    p.add_argument("--eval_every", type=int, default=10)

    # ─── Domain-shift corruption (calibrated via calibrate_domain_gap.py) ──
    p.add_argument("--use_corruption", type=int, default=0, choices=[0, 1],
                   help="1 = apply domain-shift corruption to the JEPA context "
                        "view during training; 0 = standard JEPA (no corruption).")
    p.add_argument("--corruption_prob", type=float, default=0.5)
    p.add_argument("--corruption_mode", default="single", choices=["single", "mixed"])
    p.add_argument("--mix_prob", type=float, default=0.4)
    p.add_argument("--color_temp_strength", type=float, default=0.1)
    p.add_argument("--gamma_strength", type=float, default=0.3)
    p.add_argument("--channel_mix_strength", type=float, default=0.1)
    p.add_argument("--desaturate_strength", type=float, default=0.25)
    p.add_argument("--blur_sigma_max", type=float, default=1.0)
    p.add_argument("--corruption_std", type=float, default=0.01)
    p.add_argument("--vignette_strength", type=float, default=0.05)

    # ─── Gabor filter bank (shared by A1 and A2) ──────────────
    p.add_argument("--gabor_orient", type=int, default=8,
        help="Number of orientations (theta = pi*o/gabor_orient).")
    p.add_argument("--gabor_gray", type=int, default=1, choices=[0, 1],
        help="1 = collapse to grayscale (correct for CASIA). "
             "0 = per-channel Gabor (RGB datasets, e.g. XJTU).")
    p.add_argument("--gabor_num_scales", type=int, default=3,
        choices=list(range(1, len(BASE_SCALE_LADDER) + 1)),
        help=f"Number of scales drawn from BASE_SCALE_LADDER in gabor.py "
             f"(finest -> coarsest, {len(BASE_SCALE_LADDER)} available). "
             f"Ignored if --gabor_scales is given explicitly. Default 3 "
             f"= {BASE_SCALE_LADDER[:3]}. NOTE: kernel sizes beyond "
             f"~13px exceed the default patch size "
             f"(img_size/num_patches = 14px) so filters start bleeding "
             f"across patch boundaries -- re-check if you use "
             f"--gabor_num_scales > 3.")
    p.add_argument("--gabor_gamma", type=float, default=0.5,
        help="Gaussian envelope aspect ratio for every Gabor kernel. "
             "<1 elongates filters along the orientation axis "
             "(more line-selective); 1.0 = circular envelope.")
    p.add_argument("--gabor_scales", type=str, default=None,
        help="Optional JSON list of [kernel_size, sigma, lambda] triples "
             "that OVERRIDES --gabor_num_scales when given, e.g. "
             f"'{json.dumps([list(s) for s in BASE_SCALE_LADDER[:3]])}'. "
             "Leave unset to build the bank from BASE_SCALE_LADDER via "
             "--gabor_num_scales.")
    p.add_argument("--gabor_log_every", type=int, default=5,
        help="Print structural/diagnostic logs every N epochs.")

    # ─── Legacy A1 aliases (deprecated; prefer --struct_mode) ──
    p.add_argument("--use_gabor", type=int, default=0, choices=[0, 1],
                   help="DEPRECATED alias for --struct_mode a1. "
                        "Prefer --struct_mode.")
    p.add_argument("--gabor_weight", type=float, default=0.3,
                   help="DEPRECATED alias for --w_a1 (only read when "
                        "--use_gabor 1 resolves struct_mode to 'a1').")
    p.add_argument("--gabor_schedule", default="constant",
                   choices=["constant", "decay", "ramp", "cosine"],
                   help="Only applies to the legacy --use_gabor path. "
                        "Decay was shown to underperform constant.")
    p.add_argument("--gabor_weight_final", type=float, default=0.0)
    p.add_argument("--gabor_schedule_end", type=float, default=0.25,
                   help="Fraction of total epochs the ramp/decay spans.")

    # ─── Structural task selection (A1 / A2) ──────────────────
    p.add_argument("--struct_mode", default="none",
                   choices=["none", "a1", "a2", "both"],
                   help="a1 = structure on visible patches (via ctx_embeds). "
                        "a2 = structure on hidden patches (via predictor). "
                        "both = A1+A2 sharing (or not, see struct_head_mode) "
                        "one structure head.")
    p.add_argument("--w_a1", type=float, default=0.3)
    p.add_argument("--w_a2", type=float, default=0.3)
    p.add_argument("--struct_head_hidden", type=int, default=128)
    p.add_argument("--struct_head_mode", default="shared",
                   choices=["shared", "separate"],
                   help="shared = one StructureHead for A1 and A2. "
                        "separate = independent heads (only differs from "
                        "'shared' when --struct_mode both).")
    p.add_argument("--norm_struct_out", type=int, default=1, choices=[0, 1],
                   help="LayerNorm on the predictor's structure output "
                        "(out_proj_struct) so its distribution matches "
                        "ctx_embeds. Matters most when struct_head_mode=shared, "
                        "since an unnormalized A2 output otherwise feeds the "
                        "shared head a differently-scaled input than A1's "
                        "ctx_embeds. Changes the architecture: pass "
                        "--norm_struct_out 0 to reproduce earlier checkpoints.")

    # ─── Structural loss form ─────────────────────────────────
    p.add_argument("--struct_loss", default="cosine",
                   choices=["cosine", "infonce", "smooth_l1"],
                   help="Default structural loss for both tasks.")
    p.add_argument("--struct_loss_a1", default=None,
                   choices=["cosine", "infonce", "smooth_l1"],
                   help="Override --struct_loss for the A1 (visible) task.")
    p.add_argument("--struct_loss_a2", default=None,
                   choices=["cosine", "infonce", "smooth_l1"],
                   help="Override --struct_loss for the A2 (hidden) task.")
    p.add_argument("--infonce_temp", type=float, default=0.1)
    p.add_argument("--infonce_max_n", type=int, default=4096)

    # ─── Task weighting ───────────────────────────────────────
    p.add_argument("--task_weighting", default="fixed",
                   choices=["fixed", "uncertainty"])

    # ─── Diagnostics ──────────────────────────────────────────
    p.add_argument("--log_conflict", type=int, default=1, choices=[0, 1],
                   help="Log cosine between task gradients on shared params.")

    # ─── Supervised identity term (uses source-domain labels) ──
    p.add_argument("--use_supervision", type=int, default=0, choices=[0, 1],
                   help="1 = add a supervised identity loss on pooled "
                        "context embeddings. Makes the method semi-supervised.")
    p.add_argument("--sup_loss", default="supcon",
                   choices=["supcon", "arcface", "ce"],
                   help="supcon = prototype-free metric loss (generalizes to "
                        "unseen IDs). arcface = angular-margin softmax. "
                        "ce = plain cross-entropy (closed-set baseline arm).")
    p.add_argument("--w_sup", type=float, default=0.1,
                   help="Weight of the supervised term. Keep small (0.05-0.2) "
                        "or the method becomes supervised with a JEPA regulariser.")
    p.add_argument("--supcon_temp", type=float, default=0.1)
    p.add_argument("--arcface_scale", type=float, default=30.0)
    p.add_argument("--arcface_margin", type=float, default=0.5)

    # ─── C-JEPA regularizer (Mo & Tong, NeurIPS 2024, arXiv:2410.19560) ──
    p.add_argument("--use_cjepa_reg", type=int, default=0, choices=[0, 1],
        help="1 = add the C-JEPA pairwise variance-invariance-covariance "
             "regularizer across the M target-block predictions. No extra "
             "augmented views or forward passes -- reuses the existing "
             "predictor output. Requires --num_blocks >= 2.")
    p.add_argument("--cjepa_weight", type=float, default=0.001,
        help="Outer scale on the C-JEPA term (paper's beta_vicreg).")
    p.add_argument("--cjepa_sim_weight", type=float, default=25.0,
        help="Invariance (MSE) weight inside the C-JEPA term (beta_sim).")
    p.add_argument("--cjepa_std_weight", type=float, default=25.0,
        help="Variance/anti-collapse weight inside the C-JEPA term (beta_std).")
    p.add_argument("--cjepa_cov_weight", type=float, default=1.0,
        help="Covariance/decorrelation weight inside the C-JEPA term (beta_cov).")
    p.add_argument("--cjepa_gamma", type=float, default=1.0)
    p.add_argument("--cjepa_eps", type=float, default=1e-4)
    p.add_argument("--cjepa_proj_dim", type=int, default=None,
        help="C-JEPA projector output dim. Defaults to --embed_dim.")
    p.add_argument("--cjepa_proj_hidden", type=int, default=None,
        help="C-JEPA projector hidden dim. Defaults to --embed_dim.")

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_jepa")

    cfg = p.parse_args(args)

    # ─── Resolve legacy --use_gabor into --struct_mode ────────
    # --use_gabor 1 (with struct_mode left at "none") == --struct_mode a1,
    # so existing A1 command lines keep working unchanged.
    if cfg.struct_mode == "none" and int(cfg.use_gabor) == 1:
        cfg.struct_mode = "a1"
        cfg.w_a1 = cfg.gabor_weight
    cfg.use_a1 = cfg.struct_mode in ("a1", "both")
    cfg.use_a2 = cfg.struct_mode in ("a2", "both")
    cfg.loss_a1 = cfg.struct_loss_a1 or cfg.struct_loss
    cfg.loss_a2 = cfg.struct_loss_a2 or cfg.struct_loss

    # --gabor_scales arrives as an optional JSON string (argparse can't take
    # nested tuples directly). Explicit value wins; otherwise slice
    # BASE_SCALE_LADDER by --gabor_num_scales.
    if cfg.gabor_scales is not None:
        cfg.gabor_scales = tuple(tuple(s) for s in json.loads(cfg.gabor_scales))
    else:
        cfg.gabor_scales = BASE_SCALE_LADDER[:cfg.gabor_num_scales]

    return cfg
