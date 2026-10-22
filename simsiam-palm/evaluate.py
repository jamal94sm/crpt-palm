"""
evaluate.py — Rank-1 + EER evaluation for JEPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve


@torch.no_grad()
def extract_features(feature_extractor, loader, device):
    """Extract features and labels from a dataloader."""
    feats, labels = [], []
    feature_extractor.eval()
    for x, y in loader:
        z = feature_extractor(x.to(device))
        z = F.normalize(z, dim=-1)
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def compute_eer(genuine_scores, impostor_scores):
    """Compute Equal Error Rate."""
    labels = np.concatenate([np.ones(len(genuine_scores)),
                             np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2) * 100


def evaluate_rank1_eer(gallery_feats, gallery_labels,
                       probe_feats, probe_labels):
    """Compute Rank-1 accuracy and EER from gallery/probe features."""
    sim = probe_feats @ gallery_feats.T

    # Rank-1
    top_idx = sim.argmax(dim=1)
    predicted = gallery_labels[top_idx]
    rank1 = (predicted == probe_labels).float().mean().item() * 100

    # EER
    genuine, impostor = [], []
    for i in range(len(probe_labels)):
        pid = probe_labels[i].item()
        sims = sim[i].numpy()
        glabs = gallery_labels.numpy()
        gen_mask = glabs == pid
        imp_mask = glabs != pid
        if gen_mask.any():
            genuine.extend(sims[gen_mask].tolist())
        if imp_mask.any():
            impostor.extend(sims[imp_mask].tolist())

    eer = compute_eer(np.array(genuine), np.array(impostor))

    return {
        "rank1": rank1,
        "eer": eer,
        "n_gallery": len(gallery_labels),
        "n_probe": len(probe_labels),
    }


def train_linear_probe(feats, labels, cfg):
    """Two-layer probe on FROZEN (L2-normalised) features:
         projection W: D -> d   (kept; its output is the new embedding)
         head       C: d -> n_adapt_ids  (used only for the softmax loss, then discarded)
    W is identity-initialised when d == D, so training starts from the zero-shot embedding.
    Hyper-parameters (--probe_*) are identical for every method and are never tuned on the test data."""
    D = feats.size(1)
    d = int(getattr(cfg, "probe_dim", 0) or D)
    uniq, y = torch.unique(labels, return_inverse=True)
    if len(uniq) < 2:
        raise SystemExit("linear probe needs >= 2 identities in the adaptation set")
    g = torch.Generator().manual_seed(int(getattr(cfg, "seed", 0)))
    W = nn.Linear(D, d, bias=False)
    C = nn.Linear(d, len(uniq))
    if d == D:
        with torch.no_grad():
            W.weight.copy_(torch.eye(D))
    opt = torch.optim.AdamW(list(W.parameters()) + list(C.parameters()),
                            lr=float(getattr(cfg, "probe_lr", 1e-3)),
                            weight_decay=float(getattr(cfg, "probe_wd", 1e-4)))
    bs = int(getattr(cfg, "probe_batch_size", 128))
    n = feats.size(0)
    for _ in range(int(getattr(cfg, "probe_epochs", 100))):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(C(W(feats[idx])), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    W.eval()
    return W


def run_full_eval(feature_extractor, eval_dict, cfg, tag=""):
    """Run Rank-1 + EER on each eval set. An entry that carries an 'adapt_loader'
    (built by dataset._probe_entry_from_samples) is evaluated with a linear probe:
    features -> probe projection W (trained on the adaptation samples) -> matching."""
    device = cfg.device
    results = {}

    for name, ev in eval_dict.items():
        gal_feats, gal_labels = extract_features(
            feature_extractor, ev["gallery_loader"], device)
        prb_feats, prb_labels = extract_features(
            feature_extractor, ev["probe_loader"], device)
        n_adapt = 0
        if ev.get("adapt_loader") is not None:
            ad_feats, ad_labels = extract_features(
                feature_extractor, ev["adapt_loader"], device)
            n_adapt = len(ad_labels)
            probe = train_linear_probe(ad_feats, ad_labels, cfg)
            with torch.no_grad():
                gal_feats = F.normalize(probe(gal_feats), dim=-1)
                prb_feats = F.normalize(probe(prb_feats), dim=-1)
        ver = evaluate_rank1_eer(gal_feats, gal_labels,
                                 prb_feats, prb_labels)

        results[name] = {
            "rank1": ver["rank1"],
            "eer": ver["eer"],
            "n_gallery": ver["n_gallery"],
            "n_probe": ver["n_probe"],
        }
        extra = f" Adapt={n_adapt}" if n_adapt else ""
        print(f"      {tag}{name}: R1={ver['rank1']:.2f}% | "
              f"EER={ver['eer']:.2f}% | "
              f"Gal={ver['n_gallery']} Prb={ver['n_probe']}{extra}")

    return results
