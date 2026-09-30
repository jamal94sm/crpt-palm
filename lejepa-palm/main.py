"""
main.py -- LeJEPA pretraining (Balestriero & LeCun, arXiv:2511.08544).
Single encoder, no predictor, no target/EMA network, no masking. Loss =
(1-lambda)*prediction + lambda*mean(SIGReg per view). Evaluation uses the
SAME encoder trained on (no separate target encoder exists to evaluate).
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
from torch.utils.data import DataLoader
from scipy import stats as scipy_stats

from config import get_cfg
from dataset import build_datasets, build_cross_dataset_eval_dict
from models import LeJepaEncoder, FeatureExtractor
from sigreg_loss import SIGReg, lejepa_prediction_loss
from multicrop_dataset import MultiCropDataset, multicrop_collate
from evaluate import run_full_eval


# ══════════════════ ecosystem plumbing (same as every other baseline) ══════════════════
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
        name = f"lejepa_{cfg.mode}_multiseed_n{getattr(cfg, 'n_runs', 3)}_seed{cfg.seed}.txt"
    else:
        name = f"lejepa_{cfg.mode}_seed{cfg.seed}.txt"
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


# ══════════════════ schedule (warmup-cosine LR; CONSTANT weight decay -- see config.py) ══
class CosineScheduler:
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


def train_lejepa(cfg, train_loader, eval_dict, out_path):
    dev = cfg.device
    D, grid = cfg.embed_dim, cfg.num_patches
    patch_px = cfg.img_size // grid
    n_local = cfg.local_crops_number
    n_global = 2
    n_views = n_global + n_local

    print(f"\n Building LeJEPA encoder (no predictor, no target/EMA network)...")
    encoder = LeJepaEncoder(
        (cfg.img_size, cfg.img_size), grid, D, cfg.vit_depth, cfg.vit_heads).to(dev)
    n_par = sum(p.numel() for p in encoder.parameters())
    print(f" Encoder: {n_par/1e6:.2f}M params | views={n_views} ({n_global} global + {n_local} local) "
          f"| lambda={cfg.lejepa_lambda} | sigreg_slices={cfg.sigreg_num_slices}")

    sigreg = SIGReg(cfg.sigreg_num_slices, cfg.sigreg_integration_bound, cfg.sigreg_num_points).to(dev)

    local_grid = max(2, round(grid * 6 / 16))
    mc_ds = MultiCropDataset(train_loader.dataset.samples, train_loader.dataset.id_map, cfg.img_size,
                             cfg.aug_multiplier, cfg.global_crops_scale, cfg.local_crops_scale,
                             n_local, local_size=local_grid * patch_px)
    loader = DataLoader(mc_ds, batch_size=cfg.lejepa_batch_size, shuffle=True, num_workers=cfg.num_workers,
                        drop_last=True, pin_memory=(dev != "cpu"), collate_fn=multicrop_collate)
    niter = len(loader)
    if niter == 0:
        raise SystemExit(f"lejepa_batch_size={cfg.lejepa_batch_size} > dataset length {len(mc_ds)}: no full batch")

    total = cfg.epochs * niter
    lr_sched = CosineScheduler(cfg.base_lr, cfg.min_lr, total, int(cfg.warmup_epochs_ratio * cfg.epochs) * niter)
    print(f" Steps/epoch={niter} total={total} peak_lr={cfg.base_lr:.2e} weight_decay={cfg.weight_decay} (constant)")

    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay,
                            betas=(cfg.adamw_beta1, cfg.adamw_beta2))
    feature_extractor = FeatureExtractor(encoder)

    print(f"\n{'─'*70}\n Training LeJEPA ({total} steps)\n{'─'*70}")
    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}
    global_step = 0

    for epoch in range(cfg.epochs):
        encoder.train()
        ep_pred = ep_sig = ep_total = 0.0
        n_bat, t0 = 0, time.time()

        for crops, _labels in loader:
            for g in opt.param_groups:
                g["lr"] = lr_sched[global_step]

            B = crops[0].shape[0]
            # Global and local crops have DIFFERENT pixel sizes (112px vs 42px) and
            # cannot be torch.cat'd into one batch -- forward each crop separately,
            # then stack into the (n_views, B, D) layout the losses expect.
            emb_list = [encoder(c.to(dev, non_blocking=True)) for c in crops]   # each (B, D)
            emb = torch.cat(emb_list, dim=0)                                    # (n_views*B, D), view-major

            l_pred, _mu = lejepa_prediction_loss(emb, n_global, n_views, B)

            emb_per_view = emb.view(n_views, B, -1)
            gen = torch.Generator(device=dev)
            gen.manual_seed(global_step)                                     # synced slice resampling per step
            l_sig = torch.stack([sigreg(emb_per_view[v], generator=gen) for v in range(n_views)]).mean()

            loss = (1 - cfg.lejepa_lambda) * l_pred + cfg.lejepa_lambda * l_sig
            if not torch.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_pred += l_pred.item()
            ep_sig += l_sig.item()
            ep_total += loss.item()
            global_step += 1
            n_bat += 1

        n = max(n_bat, 1)
        ep_loss = ep_total / n
        ep = epoch + 1
        if ep % 5 == 0 or ep == cfg.epochs or ep == 1:
            print(f" ep {ep:03d}/{cfg.epochs} loss={ep_loss:.4f} (pred={ep_pred/n:.4f} sigreg={ep_sig/n:.4f}) "
                  f"lr={lr_sched[global_step-1]:.2e} [{time.time()-t0:.1f}s]")

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

    encoder.eval()
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
    print(f"\n{'='*80}\n TRAINING COMPLETE (lejepa)\n Best epoch: {best_eval['epoch']} "
          f"(R1={best_eval['mean_rank1']:.2f}%, EER={best_eval['mean_eer']:.2f}%)\n{'='*80}")
    write_config_block(out_path, cfg, header=f"RUN CONFIG (seed={cfg.seed})")
    append_text(out_path, f"\nRESULTS -- method=lejepa mode={cfg.mode} seed={cfg.seed} "
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
        per_run.append(_flatten_entry(train_lejepa(run_cfg, train_loader, eval_dict, out_path)))
    keys = sorted(per_run[0].keys())
    summary = {k: compute_ci([r[k] for r in per_run if k in r], level=level) for k in keys}

    def _print_summary():
        print(f"\n{'='*80}\n MULTI-SEED SUMMARY ({n_runs} runs, lejepa, CI level={level:.0%})\n{'='*80}")
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
    print(f"\n{'='*80}\n LEJEPA PRETRAINING\n Mode: {cfg.mode} embed_dim={cfg.embed_dim} epochs={cfg.epochs}\n{'='*80}\n")
    out_path = resolve_output_path(cfg)
    open(out_path, "w").close()
    if bool(getattr(cfg, "use_CI", 0)):
        run_multi_seed(cfg, out_path)
        return
    train_loader, eval_dict, _id_map, _n_train_ids, _train_id_map = build_datasets(cfg)
    train_lejepa(cfg, train_loader, eval_dict, out_path)


if __name__ == "__main__":
    main()
