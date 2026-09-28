"""
main.py -- DINOv2-style pretraining (DINO CLS loss + iBOT masked-patch loss + KoLeo),
single-GPU port of facebookresearch/dinov2 train/train.py + train/ssl_meta_arch.py.
Evaluation uses the TEACHER backbone (official), on every eval split build_datasets()
produces -- including seen_dom_seen_id.
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
from models import DinoV2ViT, DINOHead, FeatureExtractor
from dino_losses import DINOLoss, iBOTPatchLoss, KoLeoLoss
from masking import make_mask_generator, build_ibot_masks
from multicrop_dataset import MultiCropDataset, multicrop_collate
from evaluate import run_full_eval


# ══════════════════ ecosystem plumbing (same as the other baselines) ══════════════════
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
        name = f"dinov2_{cfg.mode}_multiseed_n{getattr(cfg, 'n_runs', 3)}_seed{cfg.seed}.txt"
    else:
        name = f"dinov2_{cfg.mode}_seed{cfg.seed}.txt"
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
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean, "half_width": 0.0, "n": n}
    half = float(scipy_stats.t.ppf((1 + level) / 2, df=n - 1)) * std / np.sqrt(n)
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half,
            "half_width": half, "n": n}


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
        lines.append(f"{key},{s['mean']:.4f},{s['std']:.4f},{s['ci_low']:.4f},{s['ci_high']:.4f},{s['n']}")
    lines.append("SUMMARY_CSV_END")
    return "\n".join(lines) + "\n"


# ══════════════════ schedules / parameter groups (official) ══════════════════
class CosineScheduler:
    """dinov2.utils.utils.CosineScheduler: linear warm-up, then cosine base->final; constant
    `final_value` once `total_iters` is exceeded."""

    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0.0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_iters > 0 else np.zeros(0)
        n = max(total_iters - warmup_iters, 0)
        iters = np.arange(n)
        cos = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(n, 1)))
        self.schedule = np.concatenate((warmup, cos))

    def __getitem__(self, it):
        return self.final_value if it >= self.total_iters else self.schedule[it]


def _layer_id(name, n_blocks):
    """dinov2 get_vit_lr_decay_rate: embeddings/tokens -> 0, block i -> i+1, everything else -> n_blocks+1."""
    if name.startswith(("patch_embed", "pos_embed", "cls_token", "mask_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return n_blocks + 1


def build_param_groups(student, layerwise_decay, patch_embed_lr_mult):
    """Per-parameter groups with dinov2's lr_multiplier / wd_multiplier / is_last_layer:
    layer-wise LR decay on the backbone, x0.2 LR on patch_embed, no weight decay on
    biases / norms / LayerScale gammas, last layer of each head flagged for the freeze."""
    groups = []
    n_blocks = student["backbone"].n_blocks
    for mod_name, mod in student.items():
        for name, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if mod_name == "backbone":
                lr_mult = layerwise_decay ** (n_blocks + 1 - _layer_id(name, n_blocks))
            g = {"params": [p], "is_last_layer": "last_layer" in name, "lr_multiplier": lr_mult,
                 "wd_multiplier": 1.0, "name": f"{mod_name}.{name}"}
            if name.endswith(".bias") or "norm" in name or "gamma" in name:
                g["wd_multiplier"] = 0.0
            if "patch_embed" in name:
                g["lr_multiplier"] *= patch_embed_lr_mult
            groups.append(g)
    return groups


def apply_optim_scheduler(opt, lr, wd, last_layer_lr):
    for g in opt.param_groups:
        g["weight_decay"] = wd * g["wd_multiplier"]
        g["lr"] = (last_layer_lr if g["is_last_layer"] else lr) * g["lr_multiplier"]


# ══════════════════ training ══════════════════
def train_dinov2(cfg, train_loader, eval_dict, out_path):
    dev = cfg.device
    D, grid = cfg.embed_dim, cfg.num_patches
    n_tokens, patch_px = grid * grid, cfg.img_size // grid
    out_dim = cfg.dino_out_dim or D * 8
    hidden = cfg.head_hidden_dim or D * 2
    bottleneck = cfg.head_bottleneck_dim or max(D // 4, 32)
    n_local = cfg.local_crops_number
    do_ibot = cfg.ibot_loss_weight > 0
    sep_head = bool(cfg.ibot_separate_head) and do_ibot
    print(f"\n Building DINOv2 student/teacher...")

    def backbone(dp):
        return DinoV2ViT(cfg.img_size, grid, D, cfg.vit_depth, cfg.vit_heads,
                         drop_path_rate=dp, layerscale_init=cfg.layerscale_init).to(dev)

    def head():
        return DINOHead(D, out_dim, cfg.head_nlayers, hidden, bottleneck).to(dev)

    student = nn.ModuleDict({"backbone": backbone(cfg.drop_path_rate), "dino_head": head()})
    teacher = nn.ModuleDict({"backbone": backbone(0.0), "dino_head": head()})
    if sep_head:
        student["ibot_head"], teacher["ibot_head"] = head(), head()
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    n_bb = sum(p.numel() for p in student["backbone"].parameters())
    n_hd = sum(p.numel() for k, m in student.items() if k != "backbone" for p in m.parameters())
    print(f" Backbone: {n_bb/1e6:.2f}M | Heads: {n_hd/1e6:.3f}M params | prototypes={out_dim} "
          f"| local_crops={n_local} | iBOT head={'separate' if sep_head else 'shared'}")

    dino_loss = DINOLoss(out_dim, cfg.student_temp, cfg.center_momentum).to(dev)
    ibot_loss = iBOTPatchLoss(out_dim, cfg.student_temp, cfg.center_momentum).to(dev)
    koleo = KoLeoLoss()
    ibot_head_key = "ibot_head" if sep_head else "dino_head"

    # ─── data: multi-crop over the SAME training samples/labels as every other baseline ───
    local_grid = max(2, round(grid * 6 / 16))
    mc_ds = MultiCropDataset(train_loader.dataset.samples, train_loader.dataset.id_map, cfg.img_size,
                             cfg.aug_multiplier, cfg.global_crops_scale, cfg.local_crops_scale,
                             n_local, local_size=local_grid * patch_px)
    loader = DataLoader(mc_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        drop_last=True, pin_memory=(dev != "cpu"), collate_fn=multicrop_collate)
    niter = len(loader)
    if niter == 0:
        raise SystemExit(f"batch_size={cfg.batch_size} > dataset length {len(mc_ds)}: no full batch")
    mask_gen = make_mask_generator(grid)
    mask_ratio = (cfg.ibot_mask_ratio_min, cfg.ibot_mask_ratio_max)

    # ─── schedules (official train.build_schedulers) ───
    total = cfg.epochs * niter
    lr_peak = cfg.base_lr * math.sqrt(cfg.batch_size / 1024.0)          # official sqrt_wrt_1024 rule
    lr_sched = CosineScheduler(lr_peak, cfg.min_lr, total, int(cfg.warmup_epochs_ratio * cfg.epochs) * niter, 0.0)
    wd_sched = CosineScheduler(cfg.wd_start, cfg.wd_end, total)
    mom_sched = CosineScheduler(cfg.momentum_teacher, cfg.final_momentum_teacher, total)
    tt_ep = min(cfg.warmup_teacher_temp_epochs, int(0.3 * cfg.epochs))
    tt_sched = CosineScheduler(cfg.teacher_temp, cfg.teacher_temp, tt_ep * niter, tt_ep * niter,
                               cfg.warmup_teacher_temp)
    ll_sched = CosineScheduler(lr_peak, cfg.min_lr, total, int(cfg.warmup_epochs_ratio * cfg.epochs) * niter, 0.0)
    ll_sched.schedule[: min(cfg.freeze_last_layer_epochs, cfg.epochs) * niter] = 0.0
    print(f" Steps/epoch={niter}  total={total}  peak_lr={lr_peak:.2e}  teacher_temp warm-up={tt_ep} ep")

    opt = torch.optim.AdamW(build_param_groups(student, cfg.layerwise_decay, cfg.patch_embed_lr_mult),
                            lr=0.0, betas=(cfg.adamw_beta1, cfg.adamw_beta2), weight_decay=0.0)
    feature_extractor = FeatureExtractor(teacher["backbone"], use_cls=bool(cfg.eval_use_cls))

    print(f"\n{'─'*70}\n Training DINOv2 ({total} steps)\n{'─'*70}")
    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}
    n_skipped = 0

    for epoch in range(cfg.epochs):
        student.train()
        teacher.eval()
        acc = {"dino_local": 0.0, "dino_global": 0.0, "koleo": 0.0, "ibot": 0.0, "total": 0.0, "ent": 0.0}
        n_bat, t0 = 0, time.time()

        for it, (crops, _labels) in enumerate(loader):
            g_it = niter * epoch + it
            lr, wd, ll_lr = lr_sched[g_it], wd_sched[g_it], ll_sched[g_it]
            mom, t_temp = mom_sched[g_it], tt_sched[g_it]
            apply_optim_scheduler(opt, lr, wd, ll_lr)

            B = crops[0].shape[0]
            g_crops = torch.cat(crops[:2]).to(dev, non_blocking=True)                  # (2B,3,H,W) crop-major
            l_crops = torch.cat(crops[2:]).to(dev, non_blocking=True) if n_local > 0 else None
            if do_ibot:
                masks, mask_idx, mask_w = build_ibot_masks(2 * B, n_tokens, mask_ratio,
                                                           cfg.ibot_mask_sample_probability, mask_gen)
                masks, mask_idx, mask_w = masks.to(dev), mask_idx.to(dev), mask_w.to(dev)
            else:
                masks = mask_idx = mask_w = None

            # ── teacher: unmasked global crops, no grad ──
            with torch.no_grad():
                t_out = teacher["backbone"].forward_features(g_crops)
                t_cls = torch.cat(t_out["cls"].chunk(2)[::-1])                          # (B,A): A matched to B
                t_cls_head = teacher["dino_head"](t_cls)
                t_dino = dino_loss.softmax_center_teacher(t_cls_head, t_temp).view(2, B, -1)
                dino_loss.update_center(t_cls_head)
                ent = -(t_dino * torch.log(t_dino + 1e-12)).sum(-1).mean().item()
                if do_ibot and mask_idx.numel() > 0:
                    t_patch = t_out["patch"].flatten(0, 1)[mask_idx]
                    t_patch_head = teacher[ibot_head_key](t_patch)
                    t_ibot = ibot_loss.softmax_center_teacher(t_patch_head, t_temp)
                    ibot_loss.update_center(t_patch_head)

            # ── student: masked global crops + local crops ──
            s_g = student["backbone"].forward_features(g_crops, masks=masks)
            s_cls_g = s_g["cls"]
            s_cls_g_head = student["dino_head"](s_cls_g)

            n_local_terms = max(n_local * 2, 1)
            n_global_terms = (2 - 1) * 2
            denom = n_global_terms + n_local_terms
            l_dino_local = torch.zeros((), device=dev)
            if n_local > 0:
                s_l = student["backbone"].forward_features(l_crops)
                s_cls_l_head = student["dino_head"](s_l["cls"])
                l_dino_local = dino_loss(s_cls_l_head.chunk(n_local), [t_dino[0], t_dino[1]]) / denom
            l_dino_global = dino_loss([s_cls_g_head], [t_dino.flatten(0, 1)]) * 2 / denom
            l_koleo = cfg.koleo_loss_weight * sum(koleo(c) for c in s_cls_g.chunk(2))

            l_ibot = torch.zeros((), device=dev)
            if do_ibot and mask_idx.numel() > 0:
                s_patch = s_g["patch"].flatten(0, 1)[mask_idx]
                s_patch_head = student[ibot_head_key](s_patch)
                # official: * loss_scales(2) * ibot_loss_scale(1/n_global_crops=0.5) = * 1
                l_ibot = ibot_loss.forward_masked(s_patch_head, t_ibot, mask_w, 2 * B)

            loss = cfg.dino_loss_weight * (l_dino_local + l_dino_global) + l_koleo + cfg.ibot_loss_weight * l_ibot
            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad:
                for m in student.values():                                              # per-module (official)
                    nn.utils.clip_grad_norm_(m.parameters(), cfg.clip_grad)
            opt.step()

            with torch.no_grad():                                                       # teacher EMA
                for k in student.keys():
                    for ps, pt in zip(student[k].parameters(), teacher[k].parameters()):
                        pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)

            acc["dino_local"] += l_dino_local.item()
            acc["dino_global"] += l_dino_global.item()
            acc["koleo"] += l_koleo.item()
            acc["ibot"] += l_ibot.item()
            acc["total"] += loss.item()
            acc["ent"] += ent
            n_bat += 1

        n = max(n_bat, 1)
        ep_loss = acc["total"] / n
        ep = epoch + 1
        if ep % 5 == 0 or ep == cfg.epochs or ep == 1:
            print(f" ep {ep:03d}/{cfg.epochs} loss={ep_loss:.4f} (dino_l={acc['dino_local']/n:.3f} "
                  f"dino_g={acc['dino_global']/n:.3f} ibot={acc['ibot']/n:.3f} koleo={acc['koleo']/n:.3f}) "
                  f"ent={acc['ent']/n:.2f}/{math.log(out_dim):.2f} lr={lr:.2e} mom={mom:.4f} "
                  f"tT={t_temp:.3f} [{time.time()-t0:.1f}s]" + (f" skipped={n_skipped}" if n_skipped else ""))

        if ep % cfg.eval_every == 0 or ep == cfg.epochs:
            print(f"\n ── Eval at epoch {ep} ──")
            eval_results = run_full_eval(feature_extractor, eval_dict, cfg, tag=f"[ep{ep}] ")
            entry = {"epoch": ep, "loss": ep_loss}
            mean_r1 = np.mean([r["rank1"] for r in eval_results.values()])
            mean_eer = np.mean([r["eer"] for r in eval_results.values()])
            entry["mean_rank1"], entry["mean_eer"] = mean_r1, mean_eer
            entry.update(eval_results)
            eval_history.append(entry)
            if mean_eer < best_eval["mean_eer"]:
                best_eval = {"epoch": ep, "mean_rank1": mean_r1, "mean_eer": mean_eer}
                print(f" \u2605 New best EER={mean_eer:.2f}% (R1={mean_r1:.2f}%)")
            print(f" Summary: Mean R1={mean_r1:.2f}% | Mean EER={mean_eer:.2f}%\n")

    cross_dataset_results = {}
    if bool(getattr(cfg, "use_cross_dataset_eval", 0)):
        print(f"\n ── Cross-dataset evaluation (final epoch only) ──")
        cross_eval_dict = build_cross_dataset_eval_dict(cfg)
        if cross_eval_dict:
            cross_dataset_results = run_full_eval(feature_extractor, cross_eval_dict, cfg, tag="[cross-dataset] ")
            for name, r in cross_dataset_results.items():
                d = cross_eval_dict[name]
                print(f"     {name}: R1={r['rank1']:.2f}% | EER={r['eer']:.2f}% | Gal={d['n_gallery']} Prb={d['n_probe']}")
        print()

    def _print_history():
        names = list(eval_dict.keys())
        print(f"\n {'Epoch':>6} {'Loss':>8}", end="")
        for nm in names:
            print(f" │ {nm[:12]:>12} R1 EER", end="")
        print()
        for e in eval_history:
            print(f" {e['epoch']:>6} {e['loss']:>8.4f}", end="")
            for nm in names:
                if nm in e:
                    print(f" │ {e[nm]['rank1']:>6.2f} {e[nm]['eer']:>6.2f}", end="")
                else:
                    print(f" │ {'---':>6} {'---':>6}", end="")
            print()

    def _print_cross():
        if not cross_dataset_results:
            print(" (cross-dataset evaluation not run -- --use_cross_dataset_eval 0 or no dataset dirs configured)")
            return
        print(f" {'dataset':<16} {'R1':>8} {'EER':>8}")
        for nm, r in cross_dataset_results.items():
            print(f" {nm:<16} {r['rank1']:>8.2f} {r['eer']:>8.2f}")

    table_text, cross_text = capture_print(_print_history), capture_print(_print_cross)
    print(f"\n{'='*80}\n TRAINING COMPLETE (dinov2)\n Best epoch: {best_eval['epoch']} "
          f"(R1={best_eval['mean_rank1']:.2f}%, EER={best_eval['mean_eer']:.2f}%)\n{'='*80}")
    write_config_block(out_path, cfg, header=f"RUN CONFIG (seed={cfg.seed})")
    append_text(out_path, f"\nRESULTS -- method=dinov2 mode={cfg.mode} seed={cfg.seed} "
                          f"(LAST epoch = {eval_history[-1]['epoch']})\n{table_text}\n")
    append_text(out_path, f"\nCROSS-DATASET EVALUATION (final epoch only, trained on {cfg.data_dir})\n{cross_text}\n")
    print(f"\n Saved: {out_path}")
    if cross_dataset_results and eval_history:
        eval_history[-1].update(cross_dataset_results)
    return eval_history[-1] if eval_history else None


def run_multi_seed(cfg, out_path):
    n_runs = max(1, int(getattr(cfg, "n_runs", 3)))
    level = float(getattr(cfg, "ci_level", 0.95))
    if n_runs < 10:
        print(f"\n NOTE: n_runs={n_runs} < 10 -- prefer MEAN +/- STD over the CI below unless you raise --n_runs.\n")
    per_run = []
    for i in range(n_runs):
        run_cfg = copy.copy(cfg)
        run_cfg.seed = cfg.seed + i
        run_cfg.use_CI = 0
        print(f"\n{'─'*80}\n RUN {i+1}/{n_runs} (seed={run_cfg.seed})\n{'─'*80}")
        set_seed(run_cfg.seed)
        train_loader, eval_dict, _id_map, _n_train_ids, _train_id_map = build_datasets(run_cfg)
        per_run.append(_flatten_entry(train_dinov2(run_cfg, train_loader, eval_dict, out_path)))
    keys = sorted(per_run[0].keys())
    summary = {k: compute_ci([r[k] for r in per_run if k in r], level=level) for k in keys}

    def _print_summary():
        print(f"\n{'='*80}\n MULTI-SEED SUMMARY ({n_runs} runs, dinov2, CI level={level:.0%})\n{'='*80}")
        print(f" {'metric':<28} {'mean':>8} {'std':>8} {'CI low':>8} {'CI high':>8}")
        for k in keys:
            s = summary[k]
            print(f" {k:<28} {s['mean']:>8.3f} {s['std']:>8.3f} {s['ci_low']:>8.3f} {s['ci_high']:>8.3f}")
        print(f"{'='*80}\n")

    append_text(out_path, capture_print(_print_summary))
    append_text(out_path, _csv_block(keys, summary))
    print(f" Saved: {out_path}\n")
    return summary


def main():
    cfg = get_cfg()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"\n{'='*80}\n DINOv2 PRETRAINING\n Mode: {cfg.mode} embed_dim={cfg.embed_dim} epochs={cfg.epochs}\n{'='*80}\n")
    out_path = resolve_output_path(cfg)
    open(out_path, "w").close()
    if bool(getattr(cfg, "use_CI", 0)):
        run_multi_seed(cfg, out_path)
        return
    train_loader, eval_dict, _id_map, _n_train_ids, _train_id_map = build_datasets(cfg)
    train_dinov2(cfg, train_loader, eval_dict, out_path)


if __name__ == "__main__":
    main()
