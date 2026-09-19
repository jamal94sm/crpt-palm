"""
main.py — MAE pretraining on the same palmprint dataset/config family as
crpt-palm's proposed method. Verified against facebookresearch/mae's
models_mae.py (masking/encoder/decoder mechanics, fetched via search) and
PRETRAIN.md (optimizer/schedule, fetched directly).
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import copy
import math
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

from config import get_cfg
from dataset import build_datasets, build_cross_dataset_eval_dict
from models import MAEEncoder, MAEDecoder, FeatureExtractor
from patchify import patchify, normalize_pixel_target
from evaluate import run_full_eval


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_output_path(cfg):
    if getattr(cfg, "output_name", None):
        name = cfg.output_name
    elif getattr(cfg, "use_CI", 0):
        name = f"mae_{cfg.mode}_multiseed_n{getattr(cfg, 'n_runs', 3)}_seed{cfg.seed}.txt"
    else:
        name = f"mae_{cfg.mode}_seed{cfg.seed}.txt"
    return os.path.join(cfg.output_dir, name)


def write_config_block(path, cfg, header=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(f"\n{'='*70}\n{header or 'RUN CONFIG'}\n{'='*70}\n")
        for k in sorted(vars(cfg)):
            f.write(f"{k}: {getattr(cfg, k)}\n")
        f.write("\n")


def capture_print(fn, *args, **kwargs):
    buf = io.StringIO()
    real_stdout = sys.stdout

    class _Tee:
        def write(self, s):
            real_stdout.write(s)
            buf.write(s)

        def flush(self):
            real_stdout.flush()

    sys.stdout = _Tee()
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = real_stdout
    return buf.getvalue()


def append_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(text)


def compute_ci(values, level=0.95):
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean,
                "half_width": 0.0, "n": n}
    sem = std / np.sqrt(n)
    tval = float(scipy_stats.t.ppf((1 + level) / 2, df=n - 1))
    half = tval * sem
    return {"mean": mean, "std": std, "ci_low": mean - half,
            "ci_high": mean + half, "half_width": half, "n": n}


def _flatten_entry(entry):
    out = {"mean_rank1": entry.get("mean_rank1"), "mean_eer": entry.get("mean_eer")}
    for key, val in entry.items():
        if isinstance(val, dict) and "rank1" in val and "eer" in val:
            out[f"{key}__rank1"] = val["rank1"]
            out[f"{key}__eer"] = val["eer"]
    return out


def _csv_block(metric_keys, summary):
    lines = ["SUMMARY_CSV_START", "metric,mean,std,ci_low,ci_high,n"]
    for key in metric_keys:
        s = summary[key]
        lines.append(f"{key},{s['mean']:.4f},{s['std']:.4f},"
                      f"{s['ci_low']:.4f},{s['ci_high']:.4f},{s['n']}")
    lines.append("SUMMARY_CSV_END")
    return "\n".join(lines) + "\n"


def build_no_decay_param_groups(model, wd):
    """Same convention as maskfeat-palm: no weight decay on 1D params
    (bias, LayerNorm) and mask_token."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or "mask_token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_mae(cfg, train_loader, eval_dict, out_path):
    img_size = (cfg.img_size, cfg.img_size)
    patch_size = cfg.img_size // cfg.num_patches
    patch_pixel_dim = patch_size * patch_size * 3

    print(f"\n Building MAE encoder+decoder...")
    encoder = MAEEncoder(img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    decoder = MAEDecoder(cfg.num_patches, cfg.embed_dim, cfg.decoder_dim,
                         cfg.decoder_depth, cfg.decoder_heads,
                         patch_pixel_dim=patch_pixel_dim).to(cfg.device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f" Encoder: {n_enc/1e6:.2f}M | Decoder: {n_dec/1e6:.2f}M params | "
          f"patch_pixel_dim={patch_pixel_dim}")
    print(f" Mask ratio: {cfg.mask_ratio}  (random, official)  "
          f"norm_pix_loss: {'ON' if cfg.norm_pix_loss else 'OFF'}")

    param_groups = build_no_decay_param_groups(encoder, cfg.adamw_wd) + \
                   build_no_decay_param_groups(decoder, cfg.adamw_wd)
    base_lr = cfg.base_lr * cfg.batch_size / 256
    opt = torch.optim.AdamW(param_groups, lr=base_lr,
                            betas=(cfg.adamw_beta1, cfg.adamw_beta2))

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_epochs_ratio * total_steps)

    def adjust_lr(step):
        if step < warmup_steps:
            lr = base_lr * step / max(1, warmup_steps)
        else:
            s = step - warmup_steps
            m = total_steps - warmup_steps
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * s / max(1, m)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        return lr

    feature_extractor = FeatureExtractor(encoder)

    print(f"\n{'─'*70}\n Training MAE ({total_steps} steps)\n{'─'*70}")

    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        encoder.train()
        decoder.train()

        ep_loss, n_bat = 0.0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(cfg.device)

            latent, mask, ids_restore = encoder(images, mask_ratio=cfg.mask_ratio)
            pred = decoder(latent, ids_restore)

            with torch.no_grad():
                target = patchify(images, patch_size)
                if cfg.norm_pix_loss:
                    target = normalize_pixel_target(target)

            loss_per_patch = (pred - target).pow(2).mean(dim=-1)
            loss = (loss_per_patch * mask).sum() / mask.sum()

            lr_now = adjust_lr(global_step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            n_bat += 1
            global_step += 1

        ep_loss /= max(n_bat, 1)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == cfg.epochs or epoch == 1:
            print(f" ep {epoch:03d}/{cfg.epochs} loss={ep_loss:.4f} lr={lr_now:.2e} [{elapsed:.1f}s]")

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            print(f"\n ── Eval at epoch {epoch} ──")
            encoder.eval()
            eval_results = run_full_eval(feature_extractor, eval_dict, cfg, tag=f"[ep{epoch}] ")

            eval_entry = {"epoch": epoch, "loss": ep_loss}
            mean_r1 = np.mean([r["rank1"] for r in eval_results.values()])
            mean_eer = np.mean([r["eer"] for r in eval_results.values()])
            eval_entry["mean_rank1"] = mean_r1
            eval_entry["mean_eer"] = mean_eer
            for name, r in eval_results.items():
                eval_entry[name] = r
            eval_history.append(eval_entry)

            if mean_eer < best_eval["mean_eer"]:
                best_eval = {"epoch": epoch, "mean_rank1": mean_r1, "mean_eer": mean_eer}
                print(f" \u2605 New best EER={mean_eer:.2f}% (R1={mean_r1:.2f}%)")

            print(f" Summary: Mean R1={mean_r1:.2f}% | Mean EER={mean_eer:.2f}%\n")

    encoder.eval()
    cross_dataset_results = {}
    if bool(getattr(cfg, "use_cross_dataset_eval", 0)):
        print(f"\n ── Cross-dataset evaluation (final epoch only) ──")
        cross_eval_dict = build_cross_dataset_eval_dict(cfg)
        if cross_eval_dict:
            cross_dataset_results = run_full_eval(feature_extractor, cross_eval_dict, cfg, tag="[cross-dataset] ")
            for name, r in cross_dataset_results.items():
                d = cross_eval_dict[name]
                print(f"     {name}: R1={r['rank1']:.2f}% | EER={r['eer']:.2f}% "
                      f"| Gal={d['n_gallery']} Prb={d['n_probe']}")
        print()

    def _print_history():
        eval_names = list(eval_dict.keys())
        print(f"\n {'Epoch':>6} {'Loss':>8}", end="")
        for name in eval_names:
            print(f" │ {name[:12]:>12} R1 EER", end="")
        print()
        for entry in eval_history:
            print(f" {entry['epoch']:>6} {entry['loss']:>8.4f}", end="")
            for name in eval_names:
                if name in entry:
                    r = entry[name]
                    print(f" │ {r['rank1']:>6.2f} {r['eer']:>6.2f}", end="")
                else:
                    print(f" │ {'---':>6} {'---':>6}", end="")
            print()

    def _print_cross_dataset():
        if not cross_dataset_results:
            print(" (cross-dataset evaluation not run -- "
                  "--use_cross_dataset_eval 0 or no dataset dirs configured)")
            return
        print(f" {'dataset':<16} {'R1':>8} {'EER':>8}")
        for name, r in cross_dataset_results.items():
            print(f" {name:<16} {r['rank1']:>8.2f} {r['eer']:>8.2f}")

    table_text = capture_print(_print_history)
    cross_text = capture_print(_print_cross_dataset)
    print(f"\n{'='*80}\n TRAINING COMPLETE (mae)")
    print(f" Best epoch: {best_eval['epoch']} (R1={best_eval['mean_rank1']:.2f}%, "
          f"EER={best_eval.get('mean_eer', float('nan')):.2f}%)")
    print(f"{'='*80}")

    write_config_block(out_path, cfg, header=f"RUN CONFIG (seed={cfg.seed})")
    append_text(out_path, f"\nRESULTS -- method=mae mode={cfg.mode} "
                           f"seed={cfg.seed} (LAST epoch = {eval_history[-1]['epoch']})\n"
                           f"{table_text}\n")
    append_text(out_path, f"\nCROSS-DATASET EVALUATION (final epoch only, "
                           f"trained on {cfg.data_dir})\n{cross_text}\n")
    print(f"\n Saved: {out_path}")

    if cross_dataset_results and eval_history:
        eval_history[-1].update(cross_dataset_results)

    return eval_history[-1] if eval_history else None


def run_multi_seed(cfg, out_path):
    n_runs = max(1, int(getattr(cfg, "n_runs", 3)))
    level = float(getattr(cfg, "ci_level", 0.95))
    base_seed = cfg.seed

    if n_runs < 10:
        print(f"\n NOTE: n_runs={n_runs} < 10 -- prefer MEAN +/- STD over "
              f"the CI below unless you raise --n_runs.\n")

    per_run = []
    for i in range(n_runs):
        seed = base_seed + i
        run_cfg = copy.copy(cfg)
        run_cfg.seed = seed
        run_cfg.use_CI = 0

        print(f"\n{'─'*80}\n RUN {i+1}/{n_runs} (seed={seed})\n{'─'*80}")
        set_seed(seed)
        train_loader, eval_dict, id_map, n_train_ids, train_id_map = build_datasets(run_cfg)

        final_entry = train_mae(run_cfg, train_loader, eval_dict, out_path)
        per_run.append(_flatten_entry(final_entry))

    metric_keys = sorted(per_run[0].keys())
    summary = {key: compute_ci([r[key] for r in per_run if key in r], level=level)
               for key in metric_keys}

    def _print_summary():
        print(f"\n{'='*80}\n MULTI-SEED SUMMARY ({n_runs} runs, mae, "
              f"CI level={level:.0%})\n{'='*80}")
        print(f" {'metric':<28} {'mean':>8} {'std':>8} {'CI low':>8} {'CI high':>8}")
        for key in metric_keys:
            s = summary[key]
            print(f" {key:<28} {s['mean']:>8.3f} {s['std']:>8.3f} "
                  f"{s['ci_low']:>8.3f} {s['ci_high']:>8.3f}")
        print(f"{'='*80}\n")

    summary_text = capture_print(_print_summary)
    append_text(out_path, summary_text)
    append_text(out_path, _csv_block(metric_keys, summary))
    print(f" Saved: {out_path}\n")
    return summary


def main():
    cfg = get_cfg()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n{'='*80}\n MAE PRETRAINING\n"
          f" Mode: {cfg.mode} embed_dim={cfg.embed_dim} epochs={cfg.epochs}\n{'='*80}\n")

    out_path = resolve_output_path(cfg)
    open(out_path, "w").close()

    if bool(getattr(cfg, "use_CI", 0)):
        run_multi_seed(cfg, out_path)
        return

    train_loader, eval_dict, id_map, n_train_ids, train_id_map = build_datasets(cfg)
    train_mae(cfg, train_loader, eval_dict, out_path)


if __name__ == "__main__":
    main()