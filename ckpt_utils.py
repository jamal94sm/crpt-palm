"""ckpt_utils.py -- optional saving of the INFERENCE model of selected baselines.

Enabled by --save_ckpt (root config). run_all_baselines passes it to each baseline
subprocess through the environment (SAVE_CKPT, CKPT_KEY), so the baseline configs
need no new argument.

File: {output_dir}/ckpt_{tag}_train_{train_datasets}.pth -- a plain state_dict of the
module the baseline evaluates with (target encoder for the JEPA family, ...).
"""
import os
import torch

_SAVED = set()      # (tag, train label) already saved by THIS process -> in a multi-seed run only the first seed is saved


def _norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")


def _wanted(cfg):
    names = getattr(cfg, "save_ckpt", None) or \
        [x for x in os.environ.get("SAVE_CKPT", "").split(",") if x]
    return {_norm(n) for n in names}


def _train_label(cfg):
    if getattr(cfg, "mode", None) == "cross_dataset" and getattr(cfg, "train_datasets", None):
        return "+".join(cfg.train_datasets)
    from dataset import normalize_dataset_key
    return normalize_dataset_key(cfg.data_dir) if getattr(cfg, "data_dir", None) else "unknown"


def maybe_save_ckpt(cfg, module, default_tag, skip_prefixes=()):
    """Save module.state_dict() if this baseline is listed in --save_ckpt.
    tag = $CKPT_KEY (the BASELINE_SPECS key, set by ci_utils) or default_tag."""
    wanted = _wanted(cfg)
    if not wanted:
        return None
    tag = os.environ.get("CKPT_KEY") or default_tag
    if "all" not in wanted and _norm(tag) not in wanted:
        return None
    label = _train_label(cfg)
    if (tag, label) in _SAVED:
        return None
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, f"ckpt_{tag}_train_{label}.pth")
    sd = {k: v.detach().cpu() for k, v in module.state_dict().items()
          if not k.startswith(tuple(skip_prefixes))}
    torch.save(sd, path)
    _SAVED.add((tag, label))
    print(f"  Saved inference checkpoint: {path}  ({len(sd)} tensors, seed {cfg.seed})")
    return path