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


def _opt(cfg, name, env, default=None):
    v = getattr(cfg, name, None)
    return v if v is not None else (os.environ.get(env) or default)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate a SAVED backbone (no training): --eval_ckpt_dir
# ─────────────────────────────────────────────────────────────────────────────
def find_eval_ckpt(cfg, default_tag):
    """Path of the saved backbone to evaluate, or None when --eval_ckpt_dir is not set.
    --eval_ckpt_dir (env EVAL_CKPT_DIR) is a FILE or a DIRECTORY holding ckpt_{tag}_train_{label}.pth
    (the files written by --save_ckpt). tag = $CKPT_KEY (BASELINE_SPECS key) or default_tag.
    --eval_ckpt_label (env EVAL_CKPT_LABEL) picks the train label when a directory holds several."""
    import glob
    src = _opt(cfg, "eval_ckpt_dir", "EVAL_CKPT_DIR")
    if not src:
        return None
    src = os.path.expanduser(src)
    if os.path.isfile(src):
        return src
    tag = os.environ.get("CKPT_KEY") or default_tag
    label = _opt(cfg, "eval_ckpt_label", "EVAL_CKPT_LABEL")
    hits = sorted(p for p in glob.glob(os.path.join(src, f"ckpt_{tag}_train_*.pth"))
                  if not p.endswith("_resume.pth"))
    if label:
        hits = [p for p in hits if os.path.basename(p) == f"ckpt_{tag}_train_{label}.pth"]
    if len(hits) != 1:
        have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(src, "*.pth")))
        raise SystemExit(
            f"--eval_ckpt_dir {src}: expected exactly one ckpt_{tag}_train_{label or '*'}.pth, found "
            f"{len(hits)}. Files there: {have}. Use --eval_ckpt_label <train label> to choose.")
    return hits[0]


def eval_from_ckpt(cfg, default_tag, module, feature_extractor, build_cross_fn, run_eval_fn,
                   out_path, skip_prefixes=()):
    """If --eval_ckpt_dir is set: load the saved weights into `module` (the same module that
    maybe_save_ckpt saved), run the cross-dataset evaluation (zero-shot + every --adapt_ratios
    k%) WITHOUT training, append the table to out_path and return a final-entry dict
    ({mean_rank1, mean_eer, <eval set>: {...}}) shaped like the training functions' result.
    Returns None when --eval_ckpt_dir is not set (normal training run)."""
    path = find_eval_ckpt(cfg, default_tag)
    if path is None:
        return None
    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict) or "modules" in sd:
        raise SystemExit(f"{path}: expected a plain state_dict (ckpt_*.pth, not *_resume.pth)")
    if all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    hint = (f"{path} does not match the {default_tag} backbone built from the current config "
            f"(--embed_dim / --num_patches / depth must equal the training run's)")
    try:
        res = module.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise SystemExit(f"{hint}.\n{e}")
    missing = [k for k in res.missing_keys if not k.startswith(tuple(skip_prefixes))]
    if missing or res.unexpected_keys:
        raise SystemExit(f"{hint}.\n  missing: {missing[:5]}\n  unexpected: {list(res.unexpected_keys)[:5]}")
    module.eval()
    print(f"\n{'─'*70}\n Evaluating saved backbone (no training): {path}\n{'─'*70}")
    cross = build_cross_fn(cfg)
    if not cross:
        raise SystemExit("no cross-dataset evaluation sets: set the dataset directories "
                         "(--casia_dir/--xjtu_dir/--xpalm_dir, or --mode cross_dataset with --test_datasets)")
    results = run_eval_fn(feature_extractor, cross, cfg, tag="[ckpt-eval] ")
    lines = [f" {'dataset':<24} {'R1':>8} {'EER':>8}"] + \
            [f" {n:<24} {r['rank1']:>8.2f} {r['eer']:>8.2f}" for n, r in results.items()]
    text = "\n".join(lines)
    print("\n" + text + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        f.write(f"\nEVALUATION OF SAVED BACKBONE {os.path.abspath(path)} (seed={cfg.seed}, no training)\n"
                f"{text}\n")
    import re
    zs = [r for n, r in results.items() if not re.search(r"_(enr|dis)\d+$", n)] or list(results.values())
    entry = {"mean_rank1": sum(r["rank1"] for r in zs) / len(zs),       # mean over the zero-shot sets
             "mean_eer": sum(r["eer"] for r in zs) / len(zs)}
    entry.update(results)
    return entry


class BestTracker:
    """--ckpt_select last|best, --best_metric eer|r1.
    'best' = epoch with the best mean EER (or R1) over the TRAINING dataset/domain
    eval sets (names starting with 'seen_dom'); unseen sets never influence it."""

    def __init__(self, cfg):
        self.select = _opt(cfg, "ckpt_select", "CKPT_SELECT", "last")
        self.metric = _opt(cfg, "best_metric", "BEST_METRIC", "eer")
        self.score = self.entry = self.state = None
        self.label = ""

    def _score(self, eval_results):
        keys = [k for k in eval_results if k.startswith("seen_dom")] or list(eval_results)
        if self.metric == "r1":
            return sum(eval_results[k]["rank1"] for k in keys) / len(keys)
        return -sum(eval_results[k]["eer"] for k in keys) / len(keys)      # higher = better

    def update(self, eval_results, eval_entry, module):
        """Call right after eval_history.append(...)."""
        if self.select != "best" or not eval_results:
            return
        s = self._score(eval_results)
        if self.score is None or s > self.score:
            self.score, self.entry = s, eval_entry
            self.state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            val = s if self.metric == "r1" else -s
            print(f"    ◆ best-so-far for model selection ({self.metric.upper()} on training "
                  f"set = {val:.2f}%) -> epoch {eval_entry['epoch']}")

    def finalize(self, module, eval_history):
        """After the loop, BEFORE maybe_save_ckpt / cross-dataset eval. Restores the
        chosen weights; returns the chosen eval_history entry."""
        if self.select == "best" and self.state is not None:
            module.load_state_dict(self.state)
            entry = self.entry
            self.label = f"BEST epoch = {entry['epoch']} by {self.metric.upper()} on training set"
        else:
            entry = eval_history[-1]
            self.label = f"LAST epoch = {entry['epoch']}"
        print(f"\n  Model used for saving / cross-dataset eval / reporting: {self.label}")
        return entry

    def selected_text(self, entry):
        """Small table of the chosen epoch (written to the output text file)."""
        if not any(isinstance(v, dict) and "rank1" in v for v in entry.values()):
            return "  (no training-set eval sets: unlabeled-only training; model = LAST epoch)"
        lines = [f"  {'eval set':<24} {'R1':>8} {'EER':>8}"]
        for k, v in entry.items():
            if isinstance(v, dict) and "rank1" in v:
                lines.append(f"  {k:<24} {v['rank1']:>8.2f} {v['eer']:>8.2f}")
        lines.append(f"  {'MEAN (all sets)':<24} {entry['mean_rank1']:>8.2f} {entry['mean_eer']:>8.2f}")
        return "\n".join(lines)


class Resumable:
    """Full training state for resuming: modules + optimizers + schedulers + epoch/step
    + config + RNG.  --save_resume 0|1 (default 1, saved for the baselines in --save_ckpt),
    --resume_from FILE|DIR.   `epochs_done` = epochs already trained; --epochs is the NEW
    TOTAL (saved 100 + 50 more -> --epochs 150)."""

    def __init__(self, cfg, default_tag, modules, optims=None, scheds=None):
        self.cfg = cfg
        self.tag = os.environ.get("CKPT_KEY") or default_tag
        self.modules = {k: m for k, m in modules.items() if m is not None}
        self.optims = {k: o for k, o in (optims or {}).items() if o is not None}
        self.scheds = {k: s for k, s in (scheds or {}).items() if s is not None}

    def save(self, epochs_done, global_step=0):
        import random
        import numpy as np
        cfg = self.cfg
        wanted = _wanted(cfg)
        if not wanted or ("all" not in wanted and _norm(self.tag) not in wanted):
            return None
        if str(_opt(cfg, "save_resume", "SAVE_RESUME", "1")).lower() in ("0", "false"):
            return None
        label = _train_label(cfg)
        if (self.tag, label, "resume") in _SAVED:
            return None
        os.makedirs(cfg.output_dir, exist_ok=True)
        path = os.path.join(cfg.output_dir, f"ckpt_{self.tag}_train_{label}_resume.pth")
        blob = {
            "format": 1, "tag": self.tag, "method": getattr(cfg, "method", None),
            "epoch": int(epochs_done), "global_step": int(global_step), "train_label": label,
            "config": {k: v for k, v in vars(cfg).items()
                       if isinstance(v, (int, float, str, bool, type(None), list, tuple, dict))},
            "modules": {k: {n: t.detach().cpu() for n, t in m.state_dict().items()}
                        for k, m in self.modules.items()},
            "optimizers": {k: o.state_dict() for k, o in self.optims.items()},
            "schedulers": {k: s.state_dict() for k, s in self.scheds.items()},
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        torch.save(blob, path)
        _SAVED.add((self.tag, label, "resume"))
        print(f"  Saved RESUME checkpoint: {path}  (epoch {epochs_done}, step {global_step}, "
              f"modules: {', '.join(self.modules)})")
        return path

    def _resolve(self, src):
        import glob
        src = os.path.expanduser(src)
        if os.path.isdir(src):
            hits = sorted(glob.glob(os.path.join(src, f"ckpt_{self.tag}_train_*_resume.pth")))
            if len(hits) != 1:
                raise SystemExit(f"--resume_from {src}: expected exactly one "
                                 f"ckpt_{self.tag}_train_*_resume.pth, found {len(hits)}")
            return hits[0]
        return src

    def load(self):
        """Returns (epochs_done, global_step); (0, 0) when not resuming. Call AFTER the
        models/optimizer/scheduler exist and right BEFORE the epoch loop."""
        import random
        import numpy as np
        cfg = self.cfg
        src = _opt(cfg, "resume_from", "RESUME_FROM")
        if not src:
            return 0, 0
        path = self._resolve(src)
        blob = torch.load(path, map_location="cpu", weights_only=False)   # only load files you trust
        epochs_done, step = int(blob["epoch"]), int(blob["global_step"])
        if epochs_done >= cfg.epochs:
            raise SystemExit(f"resume: checkpoint already has {epochs_done} epochs; "
                             f"set --epochs > {epochs_done} (it is the NEW TOTAL)")
        clean = True
        for k, m in self.modules.items():
            if k not in blob["modules"]:
                print(f"  [resume] WARNING module '{k}' not in checkpoint (kept freshly initialised)")
                clean = False
                continue
            sd, own = blob["modules"][k], m.state_dict()
            ok = {n: v for n, v in sd.items() if n in own and own[n].shape == v.shape}
            bad = [n for n in sd if n not in ok] + [n for n in own if n not in sd]
            m.load_state_dict(ok, strict=False)
            if bad:
                clean = False
                print(f"  [resume] WARNING '{k}': {len(bad)} tensors not loaded (shape/name "
                      f"mismatch, e.g. {bad[:3]})")
        if clean:
            try:
                for k, o in self.optims.items():
                    o.load_state_dict(blob["optimizers"][k])
                for k, s in self.scheds.items():
                    s.load_state_dict(blob["schedulers"][k])
                    s.last_epoch = step - 1          # re-apply LR for THIS run's schedule
                    s.step()
            except Exception as e:
                print(f"  [resume] WARNING optimizer/scheduler state not restored: {e}")
        else:
            print("  [resume] optimizer/scheduler state NOT restored (architecture changed)")
        try:
            r = blob["rng"]
            random.setstate(r["python"]); np.random.set_state(r["numpy"])
            torch.set_rng_state(r["torch"])
            if r.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(r["cuda"])
        except Exception:
            pass
        print(f"  [resume] loaded {path}\n           epochs_done={epochs_done} "
              f"global_step={step} -> training epochs {epochs_done + 1}..{cfg.epochs}")
        return epochs_done, step
