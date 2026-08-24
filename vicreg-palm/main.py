"""
main.py — VICReg pretraining on the same palmprint dataset/config family as
crpt-palm's proposed method, for a fair side-by-side comparison.

Uses the exact augmentation CASIADataset already applies for the shared
--aug_multiplier dataset enlargement -- no extra or different augmentation
is introduced for VICReg's invariance term (see paired_dataset.py). No
corruption module is used here; corruption.py is proposed-method-only.

python main.py --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --mode cross_domain_openset --train_spectrums WHT \
    --output_dir ./output_vicreg
"""

import os
import json
import time
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_cfg
from dataset import build_datasets
from paired_dataset import PairedCASIADataset
from models import Encoder, Expander, FeatureExtractor
from vicreg_loss import vicreg_loss
from evaluate import run_full_eval


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def ckpt_name(cfg):
    """ckpt_{dataset}_vicreg_{source_domain}.pth"""
    dataset = os.path.basename(os.path.normpath(cfg.data_dir)).lower()
    if "casia" in dataset:
        dataset = "casiams"
    elif "xjtu" in dataset:
        dataset = "xjtu"
    elif "xpalm" in dataset:
        dataset = "xpalm"
    domain = "-".join(cfg.train_spectrums) if cfg.train_spectrums else "all"
    return f"ckpt_{dataset}_vicreg_{domain}.pth"


def make_scheduler(opt, cfg, total_steps):
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return cfg.start_lr / cfg.learning_rate + \
                (1 - cfg.start_lr / cfg.learning_rate) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg.final_lr / cfg.learning_rate + \
            (1 - cfg.final_lr / cfg.learning_rate) * \
            0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def build_paired_train_loader(cfg, train_loader):
    """Rebuilds the training loader with paired augmented views, reusing the
    EXACT samples / id_map / img_size / aug_multiplier the shared
    build_datasets() already resolved -- so VICReg trains on the identical
    split, identical enlargement factor, and identical per-sample transform
    as the proposed method's context view (pre-corruption)."""
    base_ds = train_loader.dataset
    paired_ds = PairedCASIADataset(
        base_ds.samples, base_ds.id_map, cfg.img_size,
        augment=True, aug_multiplier=cfg.aug_multiplier)
    return DataLoader(paired_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, drop_last=True,
                       pin_memory=True)


def train_vicreg(cfg, train_loader, eval_dict):
    print(f"\n Building VICReg encoder...")
    encoder = Encoder((cfg.img_size, cfg.img_size), cfg.num_patches,
                       cfg.embed_dim).to(cfg.device)
    expander = Expander(cfg.embed_dim, cfg.projector_hidden_dim,
                         cfg.projector_out_dim).to(cfg.device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_exp = sum(p.numel() for p in expander.parameters())
    print(f" Encoder: {n_enc/1e6:.2f}M params")
    print(f" Expander: {n_exp/1e6:.2f}M params")
    print(f" VICReg weights: inv={cfg.vicreg_lambda_inv} "
          f"var={cfg.vicreg_lambda_var} cov={cfg.vicreg_lambda_cov} "
          f"gamma={cfg.vicreg_gamma}")

    train_loader = build_paired_train_loader(cfg, train_loader)

    train_params = list(encoder.parameters()) + list(expander.parameters())
    opt = torch.optim.AdamW(train_params, lr=cfg.learning_rate,
                             weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = make_scheduler(opt, cfg, total_steps)

    feature_extractor = FeatureExtractor(encoder)

    print(f"\n{'─'*70}")
    print(f" Training VICReg ({total_steps} steps)")
    print(f"{'─'*70}")

    global_step = 0
    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}

    for epoch in range(1, cfg.epochs + 1):
        encoder.train()
        expander.train()

        ep_loss = ep_inv = ep_var = ep_cov = 0.0
        n_bat = 0
        t0 = time.time()

        for view1, view2, labels in train_loader:
            view1 = view1.to(cfg.device)
            view2 = view2.to(cfg.device)

            z1 = expander(encoder(view1).mean(dim=1))
            z2 = expander(encoder(view2).mean(dim=1))

            loss, stats = vicreg_loss(
                z1, z2,
                lambda_inv=cfg.vicreg_lambda_inv,
                lambda_var=cfg.vicreg_lambda_var,
                lambda_cov=cfg.vicreg_lambda_cov,
                gamma=cfg.vicreg_gamma, eps=cfg.vicreg_eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            global_step += 1
            ep_loss += loss.item()
            ep_inv += stats["inv"]
            ep_var += stats["var"]
            ep_cov += stats["cov"]
            n_bat += 1

        ep_loss /= max(n_bat, 1)
        ep_inv /= max(n_bat, 1)
        ep_var /= max(n_bat, 1)
        ep_cov /= max(n_bat, 1)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == cfg.epochs or epoch == 1:
            print(f" ep {epoch:03d}/{cfg.epochs} loss={ep_loss:.4f} "
                  f"inv={ep_inv:.4f} var={ep_var:.4f} cov={ep_cov:.4f} "
                  f"lr={lr_now:.2e} [{elapsed:.1f}s]")

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            print(f"\n ── Eval at epoch {epoch} ──")
            encoder.eval()
            eval_results = run_full_eval(
                feature_extractor, eval_dict, cfg, tag=f"[ep{epoch}] ")

            eval_entry = {"epoch": epoch, "loss": ep_loss, "inv": ep_inv,
                          "var": ep_var, "cov": ep_cov}
            mean_r1 = np.mean([r["rank1"] for r in eval_results.values()])
            mean_eer = np.mean([r["eer"] for r in eval_results.values()])
            eval_entry["mean_rank1"] = mean_r1
            eval_entry["mean_eer"] = mean_eer
            for name, r in eval_results.items():
                eval_entry[name] = r
            eval_history.append(eval_entry)

            if mean_eer < best_eval["mean_eer"]:
                best_eval = {"epoch": epoch, "mean_rank1": mean_r1,
                             "mean_eer": mean_eer}
                ckpt_path = os.path.join(cfg.output_dir, ckpt_name(cfg))
                torch.save({
                    "epoch": epoch,
                    "method": "vicreg",
                    "encoder": encoder.state_dict(),
                    "expander": expander.state_dict(),
                    "arch": {"embed_dim": cfg.embed_dim,
                             "num_patches": cfg.num_patches,
                             "img_size": cfg.img_size},
                    "mean_rank1": mean_r1, "mean_eer": mean_eer,
                }, ckpt_path)
                print(f" ★ New best EER={mean_eer:.2f}% "
                      f"(R1={mean_r1:.2f}%) → saved")

            print(f" Summary: Mean R1={mean_r1:.2f}% | "
                  f"Mean EER={mean_eer:.2f}%\n")

    _print_history(eval_history, eval_dict)
    _print_footer(best_eval)

    save_path = os.path.join(cfg.output_dir, f"vicreg_{cfg.mode}_seed{cfg.seed}.json")
    with open(save_path, "w") as f:
        json.dump({
            "mode": cfg.mode, "method": "vicreg",
            "config": {
                "embed_dim": cfg.embed_dim,
                "num_patches": cfg.num_patches,
                "epochs": cfg.epochs,
                "train_spectrums": cfg.train_spectrums,
                "aug_multiplier": cfg.aug_multiplier,
                "vicreg_lambda_inv": cfg.vicreg_lambda_inv,
                "vicreg_lambda_var": cfg.vicreg_lambda_var,
                "vicreg_lambda_cov": cfg.vicreg_lambda_cov,
                "vicreg_gamma": cfg.vicreg_gamma,
            },
            "best": best_eval, "history": eval_history,
        }, f, indent=2)
    print(f"\n Saved: {save_path}")


def _print_history(eval_history, eval_dict):
    eval_names = list(eval_dict.keys())
    print(f"\n {'Epoch':>6} {'Loss':>8} {'Inv':>7} {'Var':>7} {'Cov':>7}", end="")
    for name in eval_names:
        print(f" │ {name[:12]:>12} R1 EER", end="")
    print()
    print(f" {'─'*8}{'─'*8}{'─'*7}{'─'*7}{'─'*7}", end="")
    for _ in eval_names:
        print(f"─┼─{'─'*24}", end="")
    print()
    for entry in eval_history:
        print(f" {entry['epoch']:>6} {entry['loss']:>8.4f} "
              f"{entry['inv']:>7.4f} {entry['var']:>7.4f} {entry['cov']:>7.4f}", end="")
        for name in eval_names:
            if name in entry:
                r = entry[name]
                print(f" │ {r['rank1']:>6.2f} {r['eer']:>6.2f}", end="")
            else:
                print(f" │ {'---':>6} {'---':>6}", end="")
        print()


def _print_footer(best_eval):
    print(f"\n{'='*80}")
    print(f" TRAINING COMPLETE (vicreg)")
    print(f" Best epoch: {best_eval['epoch']} "
          f"(R1={best_eval['mean_rank1']:.2f}%, "
          f"EER={best_eval.get('mean_eer', float('nan')):.2f}%)")
    print(f"{'='*80}")


def main():
    cfg = get_cfg()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f" VICREG PRETRAINING")
    print(f" Mode: {cfg.mode} embed_dim={cfg.embed_dim} "
          f"epochs={cfg.epochs} aug={cfg.aug_multiplier}×")
    print(f"{'='*80}\n")

    train_loader, eval_dict, id_map, n_train_ids, train_id_map = build_datasets(cfg)
    train_vicreg(cfg, train_loader, eval_dict)


if __name__ == "__main__":
    main()