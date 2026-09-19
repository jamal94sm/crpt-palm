"""
main.py — DINO pretraining, verified against facebookresearch/dino's
main_dino.py (fetched directly): DINOLoss, multi-crop augmentation,
EMA teacher, cosine LR/WD/momentum/teacher-temp schedules, gradient
clipping, freeze-last-layer, all reproduced with the exact official
mechanics and default values, single-GPU (no dist.all_reduce/DDP).
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
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats as scipy_stats

from config import get_cfg
from dataset import build_datasets, build_cross_dataset_eval_dict
from models import DinoViT, DINOHead, FeatureExtractor
from dino_loss import DINOLoss
from multicrop_dataset import MultiCropDataset, multicrop_collate
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
        name = f"dino_{cfg.mode}_multiseed_n{getattr(cfg, 'n_runs', 3)}_seed{cfg.seed}.txt"
    else:
        name = f"dino_{cfg.mode}_seed{cfg.seed}.txt"
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


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    """Verbatim from official utils.cosine_scheduler."""
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.linspace(0, base_value, warmup_iters) if warmup_iters > 0 else np.array([])

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_params_groups(*models):
    """Official: regularized (weight matrices) vs not (biases, 1D params)."""
    regularized, not_regularized = [], []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                not_regularized.append(param)
            else:
                regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def clip_gradients(model, clip):
    for p in model.parameters():
        if p.grad is not None:
            nn.utils.clip_grad_norm_(p, clip)


def cancel_gradients_last_layer(epoch, model, freeze_last_layer_epochs):
    if epoch >= freeze_last_layer_epochs:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n and p.grad is not None:
            p.grad = None


def train_dino(cfg, train_loader, eval_dict, out_path):
    img_size = (cfg.img_size, cfg.img_size)
    print(f"\n Building DINO student/teacher...")

    out_dim = cfg.dino_out_dim or cfg.embed_dim * 8

    student_backbone = DinoViT(img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    teacher_backbone = DinoViT(img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    student_head = DINOHead(cfg.embed_dim, out_dim, use_bn=bool(cfg.use_bn_in_head),
                            norm_last_layer=bool(cfg.norm_last_layer)).to(cfg.device)
    teacher_head = DINOHead(cfg.embed_dim, out_dim, use_bn=bool(cfg.use_bn_in_head),
                            norm_last_layer=False).to(cfg.device)

    teacher_backbone.load_state_dict(student_backbone.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    n_enc = sum(p.numel() for p in student_backbone.parameters())
    n_head = sum(p.numel() for p in student_head.parameters())
    print(f" Backbone: {n_enc/1e6:.2f}M | Head: {n_head/1e6:.3f}M params | "
          f"out_dim={out_dim} | local_crops={cfg.local_crops_number}")

    ncrops = cfg.local_crops_number + 2
    dino_loss = DINOLoss(
        out_dim, ncrops, cfg.warmup_teacher_temp, cfg.teacher_temp,
        int(cfg.warmup_teacher_temp_epochs_ratio * cfg.epochs), cfg.epochs,
        student_temp=cfg.student_temp, center_momentum=cfg.center_momentum
    ).to(cfg.device)

    train_samples = train_loader.dataset.samples
    train_id_map = train_loader.dataset.id_map
    mc_ds = MultiCropDataset(
        train_samples, train_id_map, cfg.img_size, cfg.aug_multiplier,
        tuple(cfg.global_crops_scale), tuple(cfg.local_crops_scale),
        cfg.local_crops_number)
    mc_loader = DataLoader(mc_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, drop_last=True,
                           pin_memory=True, collate_fn=multicrop_collate)

    param_groups = get_params_groups(student_backbone, student_head)
    opt = torch.optim.AdamW(param_groups)

    niter_per_ep = len(mc_loader)
    lr_schedule = cosine_scheduler(
        cfg.base_lr * cfg.batch_size / 256, cfg.min_lr,
        cfg.epochs, niter_per_ep, warmup_epochs=int(cfg.warmup_epochs_ratio * cfg.epochs))
    wd_schedule = cosine_scheduler(cfg.dino_wd_start, cfg.dino_wd_end, cfg.epochs, niter_per_ep)
    momentum_schedule = cosine_scheduler(cfg.momentum_teacher, 1.0, cfg.epochs, niter_per_ep)

    feature_extractor = FeatureExtractor(student_backbone, use_cls=bool(cfg.eval_use_cls))

    print(f"\n{'─'*70}\n Training DINO ({cfg.epochs * niter_per_ep} steps)\n{'─'*70}")

    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}

    for epoch in range(cfg.epochs):
        student_backbone.train()
        student_head.train()
        teacher_backbone.eval()
        teacher_head.eval()

        ep_loss, n_bat = 0.0, 0
        t0 = time.time()

        for it, (crops, labels) in enumerate(mc_loader):
            global_it = niter_per_ep * epoch + it
            for i, pg in enumerate(opt.param_groups):
                pg["lr"] = lr_schedule[global_it]
                if i == 0:
                    pg["weight_decay"] = wd_schedule[global_it]

            crops = [c.to(cfg.device, non_blocking=True) for c in crops]

            teacher_cls = torch.cat([teacher_backbone(c) for c in crops[:2]], dim=0)
            teacher_output = teacher_head(teacher_cls)
            student_cls = torch.cat([student_backbone(c) for c in crops], dim=0)
            student_output = student_head(student_cls)

            loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                print(f" !! Loss is {loss.item()}, skipping this step.")
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad:
                clip_gradients(student_backbone, cfg.clip_grad)
                clip_gradients(student_head, cfg.clip_grad)
            cancel_gradients_last_layer(epoch, student_head, cfg.freeze_last_layer_epochs)
            opt.step()

            with torch.no_grad():
                m = momentum_schedule[global_it]
                for pq, pk in zip(student_backbone.parameters(), teacher_backbone.parameters()):
                    pk.data.mul_(m).add_((1 - m) * pq.detach().data)
                for pq, pk in zip(student_head.parameters(), teacher_head.parameters()):
                    pk.data.mul_(m).add_((1 - m) * pq.detach().data)

            ep_loss += loss.item()
            n_bat += 1

        ep_loss /= max(n_bat, 1)
        elapsed = time.time() - t0
        epoch_disp = epoch + 1

        if epoch_disp % 5 == 0 or epoch_disp == cfg.epochs or epoch_disp == 1:
            print(f" ep {epoch_disp:03d}/{cfg.epochs} loss={ep_loss:.4f} "
                  f"lr={lr_schedule[global_it]:.2e} mom={momentum_schedule[global_it]:.4f} [{elapsed:.1f}s]")

        if epoch_disp % cfg.eval_every == 0 or epoch_disp == cfg.epochs:
            print(f"\n ── Eval at epoch {epoch_disp} ──")
            student_backbone.eval()
            eval_results = run_full_eval(feature_extractor, eval_dict, cfg, tag=f"[ep{epoch_disp}] ")

            eval_entry = {"epoch": epoch_disp, "loss": ep_loss}
            mean_r1 = np.mean([r["rank1"] for r in eval_results.values()])
            mean_eer = np.mean([r["eer"] for r in eval_results.values()])
            eval_entry["mean_rank1"] = mean_r1
            eval_entry["mean_eer"] = mean_eer
            for name, r in eval_results.items():
                eval_entry[name] = r
            eval_history.append(eval_entry)

            if mean_eer < best_eval["mean_eer"]:
                best_eval = {"epoch": epoch_disp, "mean_rank1": mean_r1, "mean_eer": mean_eer}
                print(f" \u2605 New best EER={mean_eer:.2f}% (R1={mean_r1:.2f}%)")

            print(f" Summary: Mean R1={mean_r1:.2f}% | Mean EER={mean_eer:.2f}%\n")

    student_backbone.eval()
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
    print(f"\n{'='*80}\n TRAINING COMPLETE (dino)")
    print(f" Best epoch: {best_eval['epoch']} (R1={best_eval['mean_rank1']:.2f}%, "
          f"EER={best_eval.get('mean_eer', float('nan')):.2f}%)")
    print(f"{'='*80}")

    write_config_block(out_path, cfg, header=f"RUN CONFIG (seed={cfg.seed})")
    append_text(out_path, f"\nRESULTS -- method=dino mode={cfg.mode} "
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

        final_entry = train_dino(run_cfg, train_loader, eval_dict, out_path)
        per_run.append(_flatten_entry(final_entry))

    metric_keys = sorted(per_run[0].keys())
    summary = {key: compute_ci([r[key] for r in per_run if key in r], level=level)
               for key in metric_keys}

    def _print_summary():
        print(f"\n{'='*80}\n MULTI-SEED SUMMARY ({n_runs} runs, dino, "
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

    print(f"\n{'='*80}\n DINO PRETRAINING\n"
          f" Mode: {cfg.mode} embed_dim={cfg.embed_dim} epochs={cfg.epochs}\n{'='*80}\n")

    out_path = resolve_output_path(cfg)
    open(out_path, "w").close()

    if bool(getattr(cfg, "use_CI", 0)):
        run_multi_seed(cfg, out_path)
        return

    train_loader, eval_dict, id_map, n_train_ids, train_id_map = build_datasets(cfg)
    train_dino(cfg, train_loader, eval_dict, out_path)


if __name__ == "__main__":
    main()