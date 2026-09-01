"""
dataset.py — CASIA-MS data loading + splitting for 3 JEPA modes.

Filename format: {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
Identity = subjectID_handSide (unique biometric identity)
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


ALL_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]


def parse_filename(fname):
    """Parse CASIA-MS filename → (identity, spectrum, iteration)."""
    name = os.path.splitext(fname)[0]
    parts = name.split("_")
    if len(parts) < 4:
        return None
    subject = parts[0]
    hand = parts[1]
    spectrum = parts[2]
    iteration = parts[3]
    identity = f"{subject}_{hand}"
    return identity, spectrum, iteration


def scan_dataset(data_dir):
    """Scan dataset directory → list of (filepath, identity, spectrum)."""
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        identity, spectrum, _ = parsed
        samples.append({
            "path": os.path.join(data_dir, fname),
            "identity": identity,
            "spectrum": spectrum,
        })
    return samples


def build_id_map(samples):
    """Build identity → integer label mapping."""
    ids = sorted(set(s["identity"] for s in samples))
    return {name: idx for idx, name in enumerate(ids)}


# ══════════════════════════════════════════════════════════════
#  Data splitting for 3 modes
# ══════════════════════════════════════════════════════════════

def split_mode_all(samples, test_ratio=0.2, seed=2025):
    """
    Mode 1: all domains × all IDs.
    Random sample-wise split.
    Returns: train_samples, {"all": test_samples}
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, {"all": test}


def split_mode_cross_domain(samples, train_spectrums, seed=2025, test_spectrums=None):
    """
    Mode 2: selected domains × all IDs.
    ALL training domain samples used for training (no held-out).
    Eval on each unseen domain separately.
    test_spectrums: optional explicit list restricting which unseen domains
        get evaluated (default: None = every domain not in train_spectrums,
        the original behaviour).
    Returns: train_samples, {"460": [...], "630": [...], ...}
    """
    all_spectrums = sorted(set(s["spectrum"] for s in samples))
    unseen_spectrums = [s for s in all_spectrums if s not in train_spectrums]
    if test_spectrums:
        unseen_spectrums = [s for s in unseen_spectrums if s in test_spectrums]

    train = [s for s in samples if s["spectrum"] in train_spectrums]

    eval_sets = {}
    for sp in unseen_spectrums:
        sp_samples = [s for s in samples if s["spectrum"] == sp]
        if sp_samples:
            eval_sets[sp] = sp_samples

    return train, eval_sets


def split_mode_cross_domain_openset(samples, train_spectrums,
                                     train_id_ratio=0.8, seed=2025,
                                     test_spectrums=None):
    """
    Mode 3: selected domains × selected IDs.
    ALL training domain × training ID samples used for training.
    3 evaluation sets: seen_dom×unseen_id, unseen_dom×seen_id,
                       unseen_dom×unseen_id.
    Returns: train_samples, {eval_name: eval_samples, ...}
    """
    rng = random.Random(seed)
    all_ids = sorted(set(s["identity"] for s in samples))
    rng.shuffle(all_ids)
    n_train_ids = int(len(all_ids) * train_id_ratio)
    train_ids = set(all_ids[:n_train_ids])
    unseen_ids = set(all_ids[n_train_ids:])
    # was: unseen_spectrums = [s for s in ALL_SPECTRUMS if s not in train_spectrums]
    all_spectrums = sorted(set(s["spectrum"] for s in samples))
    unseen_spectrums = [s for s in all_spectrums if s not in train_spectrums]
    if test_spectrums:
        unseen_spectrums = [s for s in unseen_spectrums if s in test_spectrums]

    # Training: ALL seen domains × seen IDs
    train = [s for s in samples
             if s["spectrum"] in train_spectrums
             and s["identity"] in train_ids]

    eval_sets = {}

    # Seen domain × unseen IDs
    seen_unseen = [s for s in samples
                   if s["spectrum"] in train_spectrums
                   and s["identity"] in unseen_ids]
    if seen_unseen:
        eval_sets["seen_dom_unseen_id"] = seen_unseen

    # Unseen domain × seen IDs
    unseen_seen = [s for s in samples
                   if s["spectrum"] in unseen_spectrums
                   and s["identity"] in train_ids]
    if unseen_seen:
        eval_sets["unseen_dom_seen_id"] = unseen_seen

    # Unseen domain × unseen IDs
    unseen_unseen = [s for s in samples
                     if s["spectrum"] in unseen_spectrums
                     and s["identity"] in unseen_ids]
    if unseen_unseen:
        eval_sets["unseen_dom_unseen_id"] = unseen_unseen

    return train, eval_sets


# ══════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ══════════════════════════════════════════════════════════════

class CASIADataset(Dataset):
    """CASIA-MS dataset with optional augmentation multiplier."""

    def __init__(self, samples, id_map, img_size=112, augment=False,
                 aug_multiplier=1):
        self.samples = samples
        self.id_map = id_map
        self.augment = augment
        self.aug_multiplier = aug_multiplier if augment else 1

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.3, 0.3, 0.1, 0.05),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.GaussianBlur(5, sigma=(0.1, 1.0)),
                ], p=0.3),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    def __len__(self):
        return len(self.samples) * self.aug_multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        img = Image.open(s["path"]).convert("RGB")
        img = self.transform(img)
        label = self.id_map[s["identity"]]
        return img, label


# ══════════════════════════════════════════════════════════════
#  Gallery / Probe splitting for verification
# ══════════════════════════════════════════════════════════════

def split_gallery_probe(samples, id_map, gallery_ratio=0.5, seed=2025):
    """
    Split samples into gallery and probe per identity.
    Returns: gallery_samples, probe_samples
    """
    rng = random.Random(seed)
    by_id = {}
    for s in samples:
        by_id.setdefault(s["identity"], []).append(s)

    gallery, probe = [], []
    for identity, id_samples in by_id.items():
        rng.shuffle(id_samples)
        n_gal = max(1, int(len(id_samples) * gallery_ratio))
        gallery.extend(id_samples[:n_gal])
        probe.extend(id_samples[n_gal:])

    return gallery, probe


# ══════════════════════════════════════════════════════════════
#  Cross-dataset evaluation (one-time, after training only)
# ══════════════════════════════════════════════════════════════

def normalize_dataset_key(data_dir):
    """CASIA-MS / XJTU-UP / X-Palm -> 'casiams' / 'xjtu' / 'xpalm', from
    the --data_dir path alone. Mirrors main.py's ckpt_name() naming."""
    name = os.path.basename(os.path.normpath(data_dir)).lower()
    if "casia" in name:
        return "casiams"
    if "xjtu" in name:
        return "xjtu"
    if "xpalm" in name:
        return "xpalm"
    return name


def scan_by_key(key, data_dir):
    """Dispatch to the right scan_* function for a dataset key."""
    if key == "xjtu":
        return scan_xjtu(data_dir)
    if key == "xpalm":
        return scan_xpalm(data_dir)
    return scan_dataset(data_dir)          # "casiams" or anything else


def build_cross_dataset_eval(cfg, other_key, other_data_dir):
    """One eval_dict-style entry (gallery_loader/probe_loader/...) for a
    dataset OTHER than the one used for training. Uses a FRESH identity
    map local to that dataset alone -- identities never overlap across
    CASIA-MS/XJTU-UP/X-Palm by construction, so there's no shared id space
    to build here, unlike the in-domain eval_dict in build_datasets().
    Same --gallery_ratio / --seed convention as in-domain eval; ALL
    samples of the foreign dataset are used (no train/test ID split --
    nothing from it was trained on)."""
    samples = scan_by_key(other_key, other_data_dir)
    id_map = build_id_map(samples)
    gal_samples, prb_samples = split_gallery_probe(
        samples, id_map, cfg.gallery_ratio, cfg.seed)

    gal_ds = CASIADataset(gal_samples, id_map, cfg.img_size, augment=False)
    prb_ds = CASIADataset(prb_samples, id_map, cfg.img_size, augment=False)
    gal_loader = DataLoader(gal_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)
    prb_loader = DataLoader(prb_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    return {
        "gallery_loader": gal_loader,
        "probe_loader": prb_loader,
        "n_samples": len(samples),
        "n_ids": len(set(s["identity"] for s in samples)),
        "n_gallery": len(gal_samples),
        "n_probe": len(prb_samples),
    }


def build_cross_dataset_eval_dict(cfg):
    """{"cross_xjtu": {...}, "cross_xpalm": {...}} -- one entry per OTHER
    dataset (not the training one) that has its root dir configured via
    --casia_dir/--xjtu_dir/--xpalm_dir. Datasets without a configured dir
    are skipped with a printed note, not an error."""
    own_key = normalize_dataset_key(cfg.data_dir)
    dir_by_key = {"casiams": getattr(cfg, "casia_dir", None),
                  "xjtu": getattr(cfg, "xjtu_dir", None),
                  "xpalm": getattr(cfg, "xpalm_dir", None)}
    flag_by_key = {"casiams": "casia_dir", "xjtu": "xjtu_dir", "xpalm": "xpalm_dir"}

    cross_eval_dict = {}
    for key, other_dir in dir_by_key.items():
        if key == own_key:
            continue                        # this IS the training set
        if not other_dir:
            print(f"      (skipping cross-dataset eval on {key}: "
                  f"--{flag_by_key[key]} not set)")
            continue
        print(f"      Scanning cross-dataset '{key}' at {other_dir} ...")
        cross_eval_dict[f"cross_{key}"] = build_cross_dataset_eval(
            cfg, key, other_dir)

    return cross_eval_dict


# ══════════════════════════════════════════════════════════════
#  Build everything for a given mode
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
#  Build everything for a given mode
# ══════════════════════════════════════════════════════════════

def build_datasets(cfg):
    """
    Returns: train_loader, eval_dict, id_map (global), n_train_ids
    """
    if "xjtu" in cfg.data_dir.lower():
        all_samples = scan_xjtu(cfg.data_dir)
    elif "xpalm" in cfg.data_dir.lower():
        all_samples = scan_xpalm(cfg.data_dir)
    else:
        all_samples = scan_dataset(cfg.data_dir)
        
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Spectrums: {sorted(set(s['spectrum'] for s in all_samples))}")
    print(f"  Identities: {len(set(s['identity'] for s in all_samples))}")

    if cfg.mode == "all":
        train_samples, eval_sets = split_mode_all(
            all_samples, cfg.test_sample_ratio, cfg.seed)
        info = "All domains × All IDs"
    elif cfg.mode == "cross_domain":
        train_samples, eval_sets = split_mode_cross_domain(
            all_samples, cfg.train_spectrums, cfg.seed,
            test_spectrums=getattr(cfg, "test_spectrums", None))
        info = f"Train domains: {cfg.train_spectrums}"
        if getattr(cfg, "test_spectrums", None):
            info += f", Test domains: {cfg.test_spectrums}"
    elif cfg.mode == "cross_domain_openset":
        train_samples, eval_sets = split_mode_cross_domain_openset(
            all_samples, cfg.train_spectrums,
            cfg.train_id_ratio, cfg.seed,
            test_spectrums=getattr(cfg, "test_spectrums", None))
        info = (f"Train domains: {cfg.train_spectrums}, "
                f"Train ID ratio: {cfg.train_id_ratio}")
        if getattr(cfg, "test_spectrums", None):
            info += f", Test domains: {cfg.test_spectrums}"

    # Global ID map from ALL samples: keeps gallery/probe labels consistent
    # across seen AND unseen identities (needed for evaluation).
    id_map = build_id_map(all_samples)
    n_classes = len(id_map)

    # Training-only ID map: CONTIGUOUS 0..K-1 over identities that actually
    # appear in the training split. This is what the CompNet CE head is sized
    # from — a classifier over unseen (open-set) IDs would be wrong.
    train_id_map = build_id_map(train_samples)
    n_train_ids = len(train_id_map)

    print(f"\n  Mode: {cfg.mode} ({info})")
    print(f"  Train samples: {len(train_samples)} "
          f"(×{cfg.aug_multiplier} aug = "
          f"{len(train_samples) * cfg.aug_multiplier})")
    print(f"  Global IDs: {n_classes}   Training IDs (CE classes): {n_train_ids}")
    for name, samples in eval_sets.items():
        n_ids = len(set(s["identity"] for s in samples))
        print(f"  Eval '{name}': {len(samples)} samples, {n_ids} IDs")

    # Training set uses the TRAINING id map (so CE labels are 0..n_train_ids-1).
    train_ds = CASIADataset(train_samples, train_id_map, cfg.img_size,
                            augment=True, aug_multiplier=cfg.aug_multiplier)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              drop_last=True, pin_memory=True)

    # Eval sets use the GLOBAL id map (seen + unseen must share label space).
    eval_dict = {}
    for name, samples in eval_sets.items():
        gal_samples, prb_samples = split_gallery_probe(
            samples, id_map, cfg.gallery_ratio, cfg.seed)
        gal_ds = CASIADataset(gal_samples, id_map, cfg.img_size, augment=False)
        prb_ds = CASIADataset(prb_samples, id_map, cfg.img_size, augment=False)
        gal_loader = DataLoader(gal_ds, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers)
        prb_loader = DataLoader(prb_ds, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers)
        eval_dict[name] = {
            "gallery_loader": gal_loader,
            "probe_loader": prb_loader,
            "n_samples": len(samples),
            "n_ids": len(set(s["identity"] for s in samples)),
            "n_gallery": len(gal_samples),
            "n_probe": len(prb_samples),
        }

    # in build_datasets, the return line:
    return train_loader, eval_dict, id_map, n_train_ids, train_id_map


#######################################################
################################## X-Palm Dataset
#########################################################
def scan_xpalm(data_root):
    """
    Scan X-Palm dataset (scanner_roi + smartphone_roi).

    Domain (spectrum) is now the FINE-GRAINED condition/color parsed from
    the filename, not a coarse scanner/smartphone label -- so any subset
    of conditions/colors can be used as --train_spectrums / --test_spectrums,
    mixed across devices if desired (e.g. train on scanner "white"+"yellow"
    plus smartphone "roll"+"wet"+"rnd").

    Scanner filenames:    {subj}_{Hand}_{color}_{iter}.jpg
      e.g. 1_Left_white_1.jpg -> domain "white"
    Smartphone filenames: {subj}_{hand}_{condition}.jpg
                       or  {subj}_{hand}_rnd_{k}.jpg  (k = 1..5)
      e.g. 1_left_roll.jpg    -> domain "roll"
           1_left_rnd_3.jpg   -> domain "rnd"  (all 5 rnd_k collapsed to one)

    Identity is shared across scanner/smartphone for the same subject+hand
    ("XPALM_{subj}_{hand}"), matching X-Palm's design as a PAIRED
    multispectral-to-smartphone dataset -- this is what makes cross-device
    train/test domain combinations meaningful for the same identities.
    """
    import os
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []
    ids = set()

    scanner_dir = os.path.join(data_root, "scanner_roi")
    if os.path.isdir(scanner_dir):
        for subj_folder in sorted(os.listdir(scanner_dir)):
            subj_dir = os.path.join(scanner_dir, subj_folder)
            if not os.path.isdir(subj_dir):
                continue
            subj_id = subj_folder
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 4:
                    continue

                hand = parts[1].lower()
                domain = parts[2].lower()          # color: white/yellow/ir/...
                identity = f"XPALM_{subj_id}_{hand}"

                samples.append({
                    "path": os.path.join(subj_dir, fname),
                    "identity": identity,
                    "spectrum": domain,
                    "device": "scanner",
                })
                ids.add(identity)

    phone_dir = os.path.join(data_root, "smartphone_roi")
    if os.path.isdir(phone_dir):
        for subj_folder in sorted(os.listdir(phone_dir)):
            subj_dir = os.path.join(phone_dir, subj_folder)
            if not os.path.isdir(subj_dir):
                continue
            subj_id = subj_folder
            for fname in sorted(os.listdir(subj_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 3:
                    continue

                hand = parts[1].lower()
                cond = parts[2].lower()
                domain = "rnd" if cond == "rnd" else cond   # collapse rnd_1..rnd_5
                identity = f"XPALM_{subj_id}_{hand}"

                samples.append({
                    "path": os.path.join(subj_dir, fname),
                    "identity": identity,
                    "spectrum": domain,
                    "device": "smartphone",
                })
                ids.add(identity)

    print(f"  [X-Palm] {len(samples)} samples, {len(ids)} identities from {data_root}")
    print(f"  [X-Palm] domains found: {sorted(set(s['spectrum'] for s in samples))}")
    return samples
    
# ========================================================
#XJTU-UP dataset
# ============================================================

def scan_xjtu(data_root):
    """Load XJTU-UP as CASIA-style sample dicts so the rest of the pipeline
       (CASIADataset, split_gallery_probe, build_id_map) consumes it unchanged.

       Directory layout:  data_root / <device> / <condition> / <id_folder> / *.img
       where <id_folder> looks like 'L_003' or 'R_012' (hand_subject).

       XJTU has no spectrum, so every sample is tagged spectrum='XJTU' and reads
       as ONE target domain. Identities are namespaced 'XJTU_<id_folder>' so they
       can never collide with CASIA identities in a shared id_map.
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
    samples = []
    ids = set()
    for device in sorted(os.listdir(data_root)):
        dev_dir = os.path.join(data_root, device)
        if not os.path.isdir(dev_dir):
            continue
        for condition in sorted(os.listdir(dev_dir)):
            cond_dir = os.path.join(dev_dir, condition)
            if not os.path.isdir(cond_dir):
                continue
            for id_folder in sorted(os.listdir(cond_dir)):
                id_dir = os.path.join(cond_dir, id_folder)
                if not os.path.isdir(id_dir):
                    continue
                identity = f"XJTU_{id_folder}"
                domain = f"{device}_{condition}"        # the XJTU "spectrum"
                for fname in sorted(os.listdir(id_dir)):
                    if fname.lower().endswith(IMG_EXTS):
                        samples.append({
                            "path": os.path.join(id_dir, fname),
                            "identity": identity,
                            "spectrum": domain,          # <- device_condition, not "XJTU"
                        })
                        ids.add(identity)
    print(f"  [XJTU] {len(samples)} samples, {len(ids)} identities from {data_root}")
    return samples


def scan_cifar10(n_images=2000, root="./cifar_cache", seed=42):
    """CIFAR-10 as CASIA-style sample dicts for a cross-DOMAIN (non-palmprint)
       probe. Each image is its own 'identity' (biometric identity is undefined
       here); spectrum tagged 'CIFAR'. Images are cached to disk as .png so the
       existing CASIADataset (which does Image.open(path)) can read them."""
    import torchvision
    os.makedirs(root, exist_ok=True)
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n_images, len(ds)))
    samples = []
    for count, i in enumerate(idxs):
        img, label = ds[i]                       # PIL image, int label
        # give each image a distinct identity so gallery/probe has >=2 per id:
        # duplicate the SAME image path into gallery+probe by using its class as
        # identity gives ~200/class; simplest is per-CLASS identity:
        ident = f"CIFAR_{label}"
        p = os.path.join(root, f"img_{i}.png")
        if not os.path.exists(p):
            img.save(p)
        samples.append({"path": p, "identity": ident, "spectrum": "CIFAR"})
    n_id = len(set(s["identity"] for s in samples))
    print(f"  [CIFAR] {len(samples)} images, {n_id} identities -> {root}")
    return samples
