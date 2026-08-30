"""ci_utils.py -- multi-seed aggregation with mean/std and confidence
intervals, for ANY method/dataset/mode this codebase supports.

Runs the SAME cfg --n_runs times (seed, seed+1, ...). Each train_* call now
returns its LAST epoch's eval_history entry (not best) and appends its own
config+results block directly to the single shared out_path text file --
this module just orchestrates the loop and appends one final summary block.

At n_runs < 10, a t-based CI is not well-justified (large t-multiplier,
unverifiable normality at that sample size) -- prefer MEAN +/- STD.
"""

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


def compute_ci(values, level=0.95):
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


def _flatten_entry(entry, eval_names):
    out = {"mean_rank1": entry.get("mean_rank1"),
           "mean_eer": entry.get("mean_eer")}
    for name in eval_names:
        r = entry.get(name)
        if r is not None:
            out[f"{name}__rank1"] = r["rank1"]
            out[f"{name}__eer"] = r["eer"]
    return out


def run_multi_seed(cfg, train_fns, out_path):
    # Local import (not at module top) to avoid a circular import: main.py
    # imports run_multi_seed from this module at load time, so this module
    # can't import FROM main.py until main.py has finished loading.
    from main import capture_print, append_text

    n_runs = max(1, int(getattr(cfg, "n_runs", 3)))
    level = float(getattr(cfg, "ci_level", 0.95))
    base_seed = cfg.seed

    if n_runs < 10:
        print(f"\n  NOTE: n_runs={n_runs} < 10 -- the t-based CI below is "
              f"not well-justified at this sample size. Prefer MEAN +/- "
              f"STD for the paper unless you raise --n_runs.\n")

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
        run_cfg.use_CI = 0

        print(f"\n{'─'*80}\n  RUN {i+1}/{n_runs}  (seed={seed})\n{'─'*80}")

        set_seed(seed)
        train_loader, eval_dict, id_map, n_train_ids, train_id_map = \
            build_datasets(run_cfg)

        if eval_names_ref is None:
            eval_names_ref = list(eval_dict.keys())

        fn = train_fns[cfg.method]
        if cfg.method == "jepa":
            final_entry = fn(run_cfg, train_loader, eval_dict, id_map,
                              n_train_ids, out_path)
        else:
            final_entry = fn(run_cfg, train_loader, eval_dict, id_map,
                              n_train_ids, train_id_map, out_path)

        per_run.append(_flatten_entry(final_entry, eval_names_ref))

    metric_keys = sorted(per_run[0].keys())
    summary = {key: compute_ci([r[key] for r in per_run if key in r], level=level)
               for key in metric_keys}

    def _print_summary():
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

    summary_text = capture_print(_print_summary)
    append_text(out_path, summary_text)
    print(f"  Saved: {out_path}\n")
    return summary
