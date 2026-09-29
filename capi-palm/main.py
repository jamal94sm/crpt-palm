"""
main.py -- CAPI pretraining (Darcet et al., TMLR 2025, arXiv:2502.08769).
Architecture and training loop verified against the official
facebookresearch/capi model.py and train_capi.py (both read in full).

Key structural facts confirmed from source (NOT reconstructed from the paper
alone): (1) teacher's OnlineClustering head is trained by ITS OWN loss via a
SEPARATE optimizer, backward()'d before the main student step; (2) SK
normalization is per-position by default (positionwise_sk=True), achieved by
transposing patch-before-head to put position first before the clustering
head sees it; (3) predictor (decoder) is cross-attention-only against the
encoder's own output, no self-attention among mask tokens; (4) teacher
momentum follows mu = 1 - lr by default, not a separate schedule.
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
from torch.utils.data import DataLoader
from scipy import stats as scipy_stats

from config import get_cfg
from dataset import build_datasets, build_cross_dataset_eval_dict
from models import CapiEncoderDecoder, FeatureExtractor
from rope_sk import OnlineClustering, L2NormLinear
from masking import collate_capi_masks
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
        name = f"capi_{cfg.mode}_multiseed_n{getattr(cfg, 'n_runs', 3)}_seed{cfg.seed}.txt"
    else:
        name = f"capi_{cfg.mode}_seed{cfg.seed}.txt"
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
            real_stdout.write(s); buf.write(s)
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
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "half_width": half, "n": n}


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


class WarmupThenCosine:
    """lr_schedule[it]: linear warmup then cosine, TRUNCATING the last
    cosine_truncation fraction (paper's own rule, 'truncate out the last
    20% of the cosine, as proposed in I-JEPA'). Truncating means the
    schedule reaches its final_value at (1 - trunc) * total_iters, then
    stays flat, rather than completing the full cosine descent."""

    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, cosine_truncation=0.0):
        self.final_value = final_value
        self.total_iters = total_iters
        effective_total = int(total_iters * (1 - cosine_truncation))
        warmup = np.linspace(0.0, base_value, warmup_iters) if warmup_iters > 0 else np.zeros(0)
        n = max(effective_total - warmup_iters, 0)
        iters = np.arange(n)
        cos = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(n, 1)))
        self.schedule = np.concatenate((warmup, cos))

    def __getitem__(self, it):
        return self.final_value if it >= len(self.schedule) else self.schedule[it]


def _layer_id(name, n_blocks):
    if name.startswith(("patch_embed", "registers", "mask_token", "freqs")):
        return 0
    if ".blocks." in name:
        return int(name.split(".blocks.")[1].split(".")[0]) + 1
    return n_blocks + 1


def build_param_groups(student_backbone, student_head, patch_embed_lr_mult):
    groups = []
    n_blocks = student_backbone.encoder.n_blocks
    for mod_name, mod in [("backbone", student_backbone), ("head", student_head)]:
        for name, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            g = {"params": [p], "lr_multiplier": 1.0, "wd_multiplier": 1.0, "name": f"{mod_name}.{name}"}
            if mod_name == "backbone" and "patch_embed" in name:
                g["lr_multiplier"] = patch_embed_lr_mult
            if name.endswith(".bias") or "norm" in name.lower() or name.endswith("freqs"):
                g["wd_multiplier"] = 0.0
            groups.append(g)
    return groups


def train_capi(cfg, train_loader, eval_dict, out_path):
    dev = cfg.device
    D, grid = cfg.embed_dim, cfg.num_patches
    n_tokens = grid * grid
    print(f"\n Building CAPI student/EMA-teacher backbones + clustering heads...")

    def backbone(dp):
        return CapiEncoderDecoder((cfg.img_size, cfg.img_size), grid, D, cfg.vit_depth, cfg.vit_heads,
                                  n_registers=cfg.n_registers, drop_path_rate=dp).to(dev)

    student_backbone = backbone(cfg.drop_path_rate)
    teacher_backbone = backbone(0.0)
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    pred_dim = student_backbone.pred_dim
    student_head = L2NormLinear(pred_dim, cfg.num_prototypes).to(dev)
    teacher_head = OnlineClustering(D, cfg.num_prototypes, n_sk_iter=cfg.n_sk_iter,
                                    target_temp=cfg.target_temp, pred_temp=cfg.pred_temp,
                                    positionwise_sk=bool(cfg.positionwise_sk)).to(dev)

    n_bb = sum(p.numel() for p in student_backbone.parameters())
    n_hd = sum(p.numel() for p in student_head.parameters())
    print(f" Backbone: {n_bb/1e6:.2f}M | Student head: {n_hd/1e6:.3f}M | prototypes={cfg.num_prototypes} "
          f"| mask_ratio={cfg.mask_ratio} pred_subsample={cfg.prediction_subsampling} | registers={cfg.n_registers}")

    loader = DataLoader(train_loader.dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=(dev != "cpu"))
    niter = len(loader)
    if niter == 0:
        raise SystemExit(f"batch_size={cfg.batch_size} > dataset length: no full batch")

    total = cfg.epochs * niter
    lr_sched = WarmupThenCosine(cfg.base_lr, cfg.min_lr, total,
                                int(cfg.warmup_epochs_ratio * cfg.epochs) * niter, cfg.cosine_truncation)
    clustering_lr_sched = WarmupThenCosine(cfg.base_lr * cfg.clustering_lr_mult, cfg.min_lr, total,
                                           int(cfg.warmup_epochs_ratio * cfg.epochs) * niter, cfg.cosine_truncation)
    print(f" Steps/epoch={niter} total={total} peak_lr={cfg.base_lr:.2e}")

    opt = torch.optim.AdamW(build_param_groups(student_backbone, student_head, cfg.patch_embed_lr_mult),
                            lr=0.0, betas=(cfg.adamw_beta1, cfg.adamw_beta2), weight_decay=0.0)
    clustering_opt = torch.optim.AdamW(teacher_head.parameters(), lr=0.0, betas=(0.9, 0.999))
    feature_extractor = FeatureExtractor(teacher_backbone)

    print(f"\n{'─'*70}\n Training CAPI ({total} steps)\n{'─'*70}")
    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}
    global_step = 0

    for epoch in range(cfg.epochs):
        student_backbone.train()
        student_head.train()
        teacher_backbone.eval()
        teacher_head.train()
        acc = {"capi": 0.0, "cluster": 0.0, "ent": 0.0}
        n_bat, t0 = 0, time.time()

        for images, _labels in loader:
            images = images.to(dev, non_blocking=True)
            B = images.size(0)
            lr = lr_sched[global_step]
            momentum = cfg.teacher_momentum if cfg.teacher_momentum is not None else max(0.0, 1.0 - lr)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_multiplier"]
                g["weight_decay"] = cfg.weight_decay * g["wd_multiplier"]
            for g in clustering_opt.param_groups:
                g["lr"] = clustering_lr_sched[global_step]

            visible_idx, predict_idx, n_predict_per_img = collate_capi_masks(
                B, grid, cfg.mask_ratio, cfg.prediction_subsampling, mask_roll=bool(cfg.mask_roll), device=dev)

            # ── teacher: full image, own clustering loss, own optimizer step ──
            with torch.no_grad():
                patch_before_head, _ = teacher_backbone.forward_pretrain(images)   # (B, n_tokens, D)
            patch_t = patch_before_head.transpose(0, 1)                            # (n_tokens, B, D): position-first
            targets_full, cluster_loss = teacher_head(patch_t)                     # targets_full: (n_tokens, B, K)
            clustering_opt.zero_grad(set_to_none=True)
            cluster_loss.backward()

            # select the n_predict targets per image (matches predict_idx's flat (B*n_tokens)-space indexing)
            targets_flat = targets_full.detach().transpose(0, 1).reshape(B * n_tokens, -1)   # (B*n_tokens, K)
            targets = targets_flat[predict_idx]                                    # (B*n_predict, K)
            target_entropy = -torch.xlogy(targets, targets).sum(-1).mean()

            # ── student: masked forward, predict targets ──
            _, backbone_pred = student_backbone.forward_pretrain(
                images, visible_indices=visible_idx, predict_indices=predict_idx, do_prediction=True)
            pred = student_head(backbone_pred)                                     # (B*n_predict, K)
            capi_loss = -torch.sum(targets * F.log_softmax(pred / cfg.pred_temp, dim=-1), dim=-1).mean()

            if not torch.isfinite(capi_loss) or not torch.isfinite(cluster_loss):
                opt.zero_grad(set_to_none=True)
                clustering_opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            capi_loss.backward()
            opt.step()
            clustering_opt.step()

            with torch.no_grad():                                                  # teacher EMA
                for ps, pt in zip(student_backbone.parameters(), teacher_backbone.parameters()):
                    pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)

            acc["capi"] += capi_loss.item()
            acc["cluster"] += cluster_loss.item()
            acc["ent"] += target_entropy.item()
            global_step += 1
            n_bat += 1

        n = max(n_bat, 1)
        ep_loss = acc["capi"] / n
        ep = epoch + 1
        if ep % 5 == 0 or ep == cfg.epochs or ep == 1:
            print(f" ep {ep:03d}/{cfg.epochs} capi_loss={ep_loss:.4f} cluster_loss={acc['cluster']/n:.4f} "
                  f"tgt_ent={acc['ent']/n:.2f}/{math.log(cfg.num_prototypes):.2f} lr={lr:.2e} mom={momentum:.4f} "
                  f"[{time.time()-t0:.1f}s]")

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

    teacher_backbone.eval()
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
    print(f"\n{'='*80}\n TRAINING COMPLETE (capi)\n Best epoch: {best_eval['epoch']} "
          f"(R1={best_eval['mean_rank1']:.2f}%, EER={best_eval['mean_eer']:.2f}%)\n{'='*80}")
    write_config_block(out_path, cfg, header=f"RUN CONFIG (seed={cfg.seed})")
    append_text(out_path, f"\nRESULTS -- method=capi mode={cfg.mode} seed={cfg.seed} "
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
        per_run.append(_flatten_entry(train_capi(run_cfg, train_loader, eval_dict, out_path)))
    keys = sorted(per_run[0].keys())
    summary = {k: compute_ci([r[k] for r in per_run if k in r], level=level) for k in keys}

    def _print_summary():
        print(f"\n{'='*80}\n MULTI-SEED SUMMARY ({n_runs} runs, capi, CI level={level:.0%})\n{'='*80}")
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
    print(f"\n{'='*80}\n CAPI PRETRAINING\n Mode: {cfg.mode} embed_dim={cfg.embed_dim} epochs={cfg.epochs}\n{'='*80}\n")
    out_path = resolve_output_path(cfg)
    open(out_path, "w").close()
    if bool(getattr(cfg, "use_CI", 0)):
        run_multi_seed(cfg, out_path)
        return
    train_loader, eval_dict, _id_map, _n_train_ids, _train_id_map = build_datasets(cfg)
    train_capi(cfg, train_loader, eval_dict, out_path)


if __name__ == "__main__":
    main()
