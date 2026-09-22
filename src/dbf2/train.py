"""Stage 1: deep spectral representation and measurement-model training.

Optimises equation (39): Bernoulli on the attenuated logit of equation (22) for the
presence targets, negative binomial for the counts, and an optional contrastive term.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import (MoleculeViewDataset, PeakBucketBatchSampler, ViewStore, collate,
                      load_targets, split_molecules)
from .model import DeepBayesFrag, FingerprintEncoder, NegBinHead, info_nce
from . import evaluate as ev


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def morgan_fingerprints(cfg: Config, n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    tgt = pd.read_parquet(cfg.paths.targets, columns=["mol_idx"])
    mol = pd.read_parquet(cfg.paths.molecules, columns=["mol_idx", "normalized_smiles"])
    smi = tgt.merge(mol, on="mol_idx", how="left").normalized_smiles
    out = np.zeros((len(smi), n_bits), dtype=np.uint8)
    for i, s in enumerate(smi):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out[i] = np.frombuffer(
                gen.GetFingerprintAsNumPy(m).astype(np.uint8).tobytes(), dtype=np.uint8)
    return out


def build_loaders(cfg: Config, presence, counts, row_of, fps=None):
    store = ViewStore(cfg.paths.views, "train")
    splits = split_molecules(cfg, cfg.data.exclude_oracle_from_training)
    if cfg.train.limit_molecules:
        rng = np.random.RandomState(cfg.train.seed)
        splits = {k: (rng.choice(v, cfg.train.limit_molecules, replace=False)
                      if len(v) > cfg.train.limit_molecules else v)
                  for k, v in splits.items()}
    common = dict(presence=presence, counts=counts, target_row=row_of,
                  max_peaks=cfg.data.max_peaks or 0, fingerprints=fps,
                  quantise_mz=cfg.model.quantise_mz,
                  quantise_merge_mode=cfg.model.quantise_merge)
    ds = {
        "train": MoleculeViewDataset(store, splits["train"],
                                     max_views=cfg.train.views_per_molecule,
                                     train=True, seed=cfg.train.seed, **common),
        "val": MoleculeViewDataset(store, splits["val"],
                                   max_views=cfg.train.views_per_molecule,
                                   train=False, seed=cfg.train.seed, **common),
        "test": MoleculeViewDataset(store, splits["test"],
                                    max_views=cfg.train.views_per_molecule,
                                    train=False, seed=cfg.train.seed, **common),
        "oracle": MoleculeViewDataset(store, splits["oracle"],
                                      max_views=cfg.train.views_per_molecule,
                                      train=False, seed=cfg.train.seed, **common),
    }
    loaders = {}
    for k, v in ds.items():
        if len(v) == 0:
            continue
        sampler = PeakBucketBatchSampler(
            v, cfg.train.batch_molecules, shuffle=(k == "train"),
            drop_last=(k == "train"), seed=cfg.train.seed)
        loaders[k] = DataLoader(v, batch_sampler=sampler, collate_fn=collate,
                                num_workers=cfg.train.num_workers,
                                pin_memory=torch.cuda.is_available())
    return store, ds, loaders


def move(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def compute_loss(cfg: Config, out, batch, fp_encoder=None) -> Dict[str, torch.Tensor]:
    losses = {}
    losses["presence"] = F.binary_cross_entropy_with_logits(
        out["logit"], batch["presence"])
    total = losses["presence"]
    if "count_log_mu" in out and "counts" in batch:
        nb = NegBinHead.nll(out["count_log_mu"], out["count_log_phi"], batch["counts"])
        losses["count"] = nb.mean()
        total = total + cfg.train.count_weight * losses["count"]
    if fp_encoder is not None and "fingerprint" in batch and "z_spectrum" in out:
        z_g = fp_encoder(batch["fingerprint"])
        losses["contrastive"] = info_nce(out["z_spectrum"], z_g, cfg.model.temperature)
        total = total + cfg.train.contrastive_weight * losses["contrastive"]
    losses["total"] = total
    return losses


@torch.no_grad()
def predict(model, loader, device, amp: bool = True):
    """Return stacked probabilities, targets, posterior variance and view counts."""
    model.eval()
    P, Y, V, NV, NVT, G = [], [], [], [], [], []
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(batch)
        P.append(torch.sigmoid(out["logit"].float()).cpu())
        V.append(out["theta_var"].float().cpu())
        if "presence" in batch:
            Y.append(batch["presence"].cpu())
        NV.append(batch["n_views"].cpu())
        NVT.append(batch["n_views_total"].cpu())
        G.append(batch["group_id"].cpu())
    return {"prob": torch.cat(P).numpy(),
            "target": torch.cat(Y).numpy() if Y else None,
            "theta_var": torch.cat(V).numpy(),
            "n_views": torch.cat(NV).numpy(),
            "n_views_total": torch.cat(NVT).numpy(),
            "group_id": torch.cat(G).numpy()}


def train(cfg: Config, out_dir: Path, tag: str = "run") -> dict:
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    presence, counts, row_of, pres_names, cnt_names = load_targets(cfg)
    cfg.model.n_presence = presence.shape[1]
    cfg.model.n_count = counts.shape[1]
    fps = morgan_fingerprints(cfg) if cfg.model.contrastive else None

    store, ds, loaders = build_loaders(cfg, presence, counts, row_of, fps)
    print(f"[train] molecules: " +
          ", ".join(f"{k}={len(v)}" for k, v in ds.items()), flush=True)

    model = DeepBayesFrag(cfg.model).to(device)
    fp_encoder = (FingerprintEncoder(fps.shape[1], 512, cfg.model.d_proj).to(device)
                  if cfg.model.contrastive else None)
    params = list(model.parameters()) + (list(fp_encoder.parameters()) if fp_encoder else [])
    n_par = sum(p.numel() for p in params if p.requires_grad)
    print(f"[train] {n_par/1e6:.2f} M parameters; device {device}", flush=True)

    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = max(len(loaders["train"]), 1) * cfg.train.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")

    history, best, bad = [], -np.inf, 0
    for epoch in range(cfg.train.epochs):
        model.train()
        if fp_encoder: fp_encoder.train()
        agg, n = {}, 0
        t0 = time.time()
        for batch in loaders["train"]:
            batch = move(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(batch)
                losses = compute_loss(cfg, out, batch, fp_encoder)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            scaler.step(opt); scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            n += 1
        tr = {k: v / max(n, 1) for k, v in agg.items()}

        val = predict(model, loaders["val"], device, cfg.train.amp)
        m = ev.presence_metrics(val["prob"], val["target"])
        rec = {"epoch": epoch, "seconds": round(time.time() - t0, 1),
               **{f"train_{k}": round(v, 5) for k, v in tr.items()},
               **{f"val_{k}": round(v, 5) for k, v in m.items()}}
        history.append(rec)
        print(f"[train] {rec}", flush=True)

        score = m["macro_auprc"]
        if score > best + 1e-5:
            best, bad = score, 0
            torch.save({"model": model.state_dict(),
                        "fp_encoder": fp_encoder.state_dict() if fp_encoder else None,
                        "cfg_model": cfg.model.__dict__, "epoch": epoch,
                        "pres_names": pres_names, "cnt_names": cnt_names},
                       out_dir / f"{tag}_best.pt")
        else:
            bad += 1
            if bad >= cfg.train.patience:
                print(f"[train] early stop at epoch {epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(out_dir / f"{tag}_history.csv", index=False)
    ck = torch.load(out_dir / f"{tag}_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    report = {"tag": tag, "best_epoch": ck["epoch"], "best_val_macro_auprc": best,
              "n_parameters": n_par, "pooling": cfg.model.pooling,
              "encoder": cfg.model.encoder}
    splits = ("val",) if getattr(cfg.train, "skip_test", False) else ("val", "test", "oracle")
    for split in splits:
        if split not in loaders:
            continue
        pr = predict(model, loaders[split], device, cfg.train.amp)
        report[split] = ev.full_report(pr)
        np.savez_compressed(out_dir / f"{tag}_{split}_predictions.npz", **pr)
    (out_dir / f"{tag}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--rung", default="M2", choices=["M0", "M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--views", type=int, default=None)
    ap.add_argument("--max-peaks", type=int, default=None)
    ap.add_argument("--contrastive", action="store_true")
    ap.add_argument("--limit-molecules", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cfg = Config.ablation(a.rung, a.root)
    if a.epochs: cfg.train.epochs = a.epochs
    if a.batch: cfg.train.batch_molecules = a.batch
    if a.views: cfg.train.views_per_molecule = a.views
    if a.max_peaks is not None: cfg.data.max_peaks = a.max_peaks
    cfg.model.contrastive = a.contrastive
    if a.limit_molecules: cfg.train.limit_molecules = a.limit_molecules
    tag = a.tag or a.rung
    out = cfg.paths.runs / tag
    cfg.to_json(out / "config.json")
    train(cfg, out, tag)


if __name__ == "__main__":
    main()
