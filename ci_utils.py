"""ci_utils.py -- multi-seed aggregation (mean/std + CI) for a single run,
and orchestration across ALL baselines under one shared condition.

run_multi_seed: unchanged mechanism from before -- runs one cfg N times,
appends a human-readable summary AND a machine-readable SUMMARY_CSV block
to the run's output file (the CSV block exists purely so run_all_baselines
can parse it reliably; the pretty-printed table is what you actually read).

run_all_baselines: launches every baseline as a SEPARATE SUBPROCESS (not an
in-process import) -- crpt-palm's own main.py and vicreg_palm/main.py have
colliding module names (config.py, dataset.py, models.py, main.py), so
importing both into one Python process risks exactly the kind of shadowing
bug this conversation already hit once with MFRL's vicreg/ subfolder.
Subprocess isolation sidesteps that entirely, at the cost of parsing text
back out of each baseline's output file instead of getting a Python object
directly.
"""

import os
import sys
import copy
import random
import subprocess

import numpy as np
import torch
from scipy import stats

from dataset import build_datasets
from config import BASELINE_SPECS, SHARED_ARG_NAMES

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


def _csv_block(metric_keys, summary):
    lines = ["SUMMARY_CSV_START", "metric,mean,std,ci_low,ci_high,n"]
    for key in metric_keys:
        s = summary[key]
        lines.append(f"{key},{s['mean']:.4f},{s['std']:.4f},"
                      f"{s['ci_low']:.4f},{s['ci_high']:.4f},{s['n']}")
    lines.append("SUMMARY_CSV_END")
    return "\n".join(lines) + "\n"


def run_multi_seed(cfg, train_fns, out_path):
    # Local import: main.py imports run_multi_seed from this module at
    # load time, so this module can only import FROM main.py after that
    # has finished loading -- hence the import lives inside the function.
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
    append_text(out_path, _csv_block(metric_keys, summary))
    print(f"  Saved: {out_path}\n")
    return summary



def _shared_flags(cfg):
    flags = []
    for name in SHARED_ARG_NAMES:
        val = getattr(cfg, name, None)
        if val is None:
            continue
        flags += [f"--{name}", str(val)]
    if getattr(cfg, "train_spectrums", None):
        flags.append("--train_spectrums")
        flags += [str(s) for s in cfg.train_spectrums]
    flags += ["--use_CI", "1", "--n_runs", str(getattr(cfg, "n_runs", 3)),
              "--ci_level", str(getattr(cfg, "ci_level", 0.95)),
              "--output_dir", os.path.abspath(cfg.output_dir)]

    if bool(getattr(cfg, "use_cross_dataset_eval", 0)):
        flags += ["--use_cross_dataset_eval", "1"]
        for name in ("casia_dir", "xjtu_dir", "xpalm_dir"):
            val = getattr(cfg, name, None)
            if val:
                flags += [f"--{name}", val]
    return flags


def _parse_summary_csv(path):
    with open(path) as f:
        text = f.read()
    start = text.rfind("SUMMARY_CSV_START")
    end = text.rfind("SUMMARY_CSV_END")
    if start == -1 or end == -1:
        raise RuntimeError(f"No SUMMARY_CSV block in {path} -- did that "
                            f"baseline finish successfully?")
    rows = text[start:end].splitlines()[2:]      # skip marker + header
    out = {}
    for line in rows:
        if not line.strip():
            continue
        key, mean, std, ci_low, ci_high, n = line.split(",")
        out[key] = {"mean": float(mean), "std": float(std),
                    "ci_low": float(ci_low), "ci_high": float(ci_high),
                    "n": int(n)}
    return out


def run_all_baselines(cfg):
    from main import write_config_block, append_text

    here = os.path.dirname(os.path.abspath(__file__))
    self_main = os.path.join(here, "main.py")
    vicreg_main = os.path.abspath(os.path.join(here, cfg.vicreg_script_path))

    os.makedirs(cfg.output_dir, exist_ok=True)
    combined_path = os.path.join(
        cfg.output_dir, cfg.combined_output_name or "ALL_BASELINES.txt")
    open(combined_path, "w").close()
    write_config_block(combined_path, cfg,
                        header="SHARED CONDITION (applies to every baseline below)")

    shared = _shared_flags(cfg)
    results = {}

    for spec in BASELINE_SPECS:
        out_name = (spec["name"].replace(" ", "_")
                    .replace("(", "").replace(")", "") + ".txt")
        if spec["script"] == "self":
            cmd = [sys.executable, self_main] + shared + spec["extra"] + \
                  ["--output_name", out_name]
            cwd = here
        else:
            cmd = [sys.executable, vicreg_main] + shared + spec["extra"] + \
                  ["--output_name", out_name]
            cwd = os.path.dirname(vicreg_main)

        print(f"\n{'#'*80}\n  BASELINE: {spec['name']}")
        print(f"  CMD: {' '.join(cmd)}\n{'#'*80}\n")
        subprocess.run(cmd, cwd=cwd, check=True)

        results[spec["name"]] = _parse_summary_csv(
            os.path.join(cfg.output_dir, out_name))

    # ─── Combined comparison table ───
    eval_names = ["seen_dom_unseen_id", "unseen_dom_seen_id", "unseen_dom_unseen_id"]
    cols = [(name, metric) for name in eval_names for metric in ("eer", "rank1")]

    header = f"{'Method':<24}" + "".join(
        f"{name[:10]}_{metric}".rjust(18) for name, metric in cols)
    lines = [header, "-" * len(header)]
    for spec in BASELINE_SPECS:
        row = f"{spec['name']:<24}"
        r = results[spec["name"]]
        for name, metric in cols:
            key = f"{name}__{metric}"
            cell = f"{r[key]['mean']:.2f} \u00b1 {r[key]['std']:.2f}" if key in r else "---"
            row += cell.rjust(18)
        lines.append(row)
    table_text = "\n".join(lines) + "\n"

    print("\n" + table_text)
    append_text(combined_path,
                f"\nCOMBINED RESULTS (mean \u00b1 std over "
                f"{getattr(cfg, 'n_runs', 3)} runs)\n")
    append_text(combined_path, table_text)
    print(f"\n  Saved combined comparison: {combined_path}\n")
