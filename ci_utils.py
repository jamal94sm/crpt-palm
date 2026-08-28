"""ci_utils.py -- multi-seed aggregation with mean/std and confidence
intervals, for ANY method/dataset/mode this codebase supports.

Runs the SAME cfg (--method, --data_dir, --mode, --train_spectrums, ...)
--n_runs times with seed, seed+1, ..., seed+n_runs-1 (full seed -- reshuffles
the unseen-ID split AND model init/training stochasticity together, the
conventional meaning of "N runs"). Each individual run's dataset split,
training loop, and per-run JSON output are UNCHANGED from a normal single
run; this module only adds an outer loop + a read-back-and-aggregate step.

--use_CI 0 (default): normal single-run behavior, this module is never
                       invoked.
--use_CI 1: orchestrates the loop below.

At n_runs < 10, a t-based CI is not well-justified (large t-multiplier,
unverifiable normality at that sample size) -- report mean +/- std instead.
This module always computes and prints BOTH, with a warning at the low end,
so you can pick whichever is right for the paper.
"""

import os
import json
import copy
import random

import numpy as np
import torch
from scipy import stats

from dataset import build_datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


_JSON_NAME = {
    "jepa": lambda mode, seed: f"jepa_{mode}_seed{seed}.json",
    "compnet": lambda mode, seed: f"compnet_{mode}_seed{seed}.json",
    "vit_sup": lambda mode, seed: f"vitsup_{mode}_seed{seed}.json",
}


def compute_ci(values, level=0.95):
    """values: list[float]. Returns dict with mean, std, ci_low, ci_high,
    half_width, n. Uses a t-distribution with n-1 df (correct for small n,
    unlike a fixed z-multiplier)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean,
                "half_width": 0.0, "n": n}
    sem = std / np.sqrt(n)
    tval = float(stats.t.ppf((1 + level) / 2, df=n - 1))
    half = tval * sem
    return {"mean": mean, "std": std, "ci_low": mean - half,
            "ci_high": mean + half, "half_width": half, "n": n}


def _read_run_metrics(cfg, seed, eval_names):
    """Load one seed's saved JSON, return per-split + mean metrics at the
    BEST epoch (matching the checkpoint that run actually saved)."""
    name_fn = _JSON_NAME.get(cfg.method)
    if name_fn is None:
        raise SystemExit(f"--use_CI aggregation not wired up for "
                          f"--method {cfg.method}")
    path = os.path.join(cfg.output_dir, name_fn(cfg.mode, seed))
    with open(path) as f:
        data = json.load(f)

    best_epoch = data["best"]["epoch"]
    entry = next((e for e in data["history"] if e["epoch"] == best_epoch),
                 data["history"][-1])

    out = {"mean_rank1": entry.get("mean_rank1", data["best"].get("mean_rank1")),
           "mean_eer": entry.get("mean_eer", data["best"].get("mean_eer"))}
    for name in eval_names:
        r = entry.get(name)
        if r is not None:
            out[f"{name}__rank1"] = r["rank1"]
            out[f"{name}__eer"] = r["eer"]
    return out


def run_multi_seed(cfg, train_fns):
    """train_fns: dict {"jepa": train_jepa, "compnet": train_compnet,
    "vit_sup": train_vit_sup} -- passed in by main.py rather than imported
    here, to avoid a circular import (main.py imports THIS module)."""
    n_runs = max(1, int(getattr(cfg, "n_runs", 3)))
    level = float(getattr(cfg, "ci_level", 0.95))
    base_seed = cfg.seed

    if n_runs < 10:
        print(f"\n  NOTE: n_runs={n_runs} < 10 -- the t-based CI below is "
              f"not well-justified at this sample size (large t-multiplier, "
              f"unverifiable normality). Prefer MEAN +/- STD for the paper "
              f"unless you raise --n_runs.\n")

    print(f"\n{'='*80}")
    print(f"  MULTI-SEED RUN  —  method: {cfg.method.upper()}   "
          f"n_runs={n_runs}   seeds={base_seed}..{base_seed + n_runs - 1}")
    print(f"{'='*80}\n")

    eval_names_ref = None
    per_run = []

    for i in range(n_runs):
        seed = base_seed + i
        run_cfg = copy.copy(cfg)
        run_cfg.seed = seed
        run_cfg.use_CI = 0                     # prevent re-entrant looping

        print(f"\n{'─'*80}\n  RUN {i+1}/{n_runs}  (seed={seed})\n{'─'*80}")

        set_seed(seed)
        train_loader, eval_dict, id_map, n_train_ids, train_id_map = \
            build_datasets(run_cfg)

        if eval_names_ref is None:
            eval_names_ref = list(eval_dict.keys())

        fn = train_fns[cfg.method]
        if cfg.method == "jepa":
            fn(run_cfg, train_loader, eval_dict, id_map, n_train_ids)
        else:
            fn(run_cfg, train_loader, eval_dict, id_map, n_train_ids, train_id_map)

        per_run.append(_read_run_metrics(run_cfg, seed, eval_names_ref))

    # ─── Aggregate ───
    metric_keys = sorted(per_run[0].keys())
    summary = {key: compute_ci([r[key] for r in per_run if key in r], level=level)
               for key in metric_keys}

    print(f"\n{'='*80}")
    print(f"  MULTI-SEED SUMMARY  ({n_runs} runs, {cfg.method}, "
          f"CI level={level:.0%})")
    print(f"{'='*80}")
    print(f"  {'metric':<28} {'mean':>8} {'std':>8} {'CI low':>8} {'CI high':>8}")
    for key in metric_keys:
        s = summary[key]
        print(f"  {key:<28} {s['mean']:>8.3f} {s['std']:>8.3f} "
              f"{s['ci_low']:>8.3f} {s['ci_high']:>8.3f}")
    print(f"{'='*80}\n")

    out_path = os.path.join(
        cfg.output_dir,
        f"multiseed_{cfg.method}_{cfg.mode}_n{n_runs}_seed{base_seed}.json")
    with open(out_path, "w") as f:
        json.dump({
            "method": cfg.method, "mode": cfg.mode, "n_runs": n_runs,
            "seeds": list(range(base_seed, base_seed + n_runs)),
            "ci_level": level,
            "per_run": per_run,
            "summary": summary,
        }, f, indent=2)
    print(f"  Saved: {out_path}\n")
    return summary