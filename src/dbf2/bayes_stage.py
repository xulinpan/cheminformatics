"""Stage 2 driver: freeze the encoder, extract sufficient statistics, run the
Polya-Gamma BART sampler, and measure whether the Bayesian layer earns its cost.

This is ablation rung M3 against M2 (specification section 15).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import MoleculeViewDataset, ViewStore, collate, load_targets, split_molecules
from .measurement import MeasurementModel
from .model import DeepBayesFrag
from .pg_bart import PGBARTSampler, posterior_presence, HAVE_POLYAGAMMA
from . import evaluate as ev


@torch.no_grad()
def extract_statistics(model: DeepBayesFrag, loader: DataLoader, device
                       ) -> Dict[str, np.ndarray]:
    """H_pool, the prior offset a_ij, and the view sufficient statistics s1, s2."""
    model.eval()
    H, A, S1, S2, Y, G, NV = [], [], [], [], [], [], []
    hierarchical = isinstance(model.aggregator, MeasurementModel)
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)
        H.append(out["h_pool"].float().cpu())
        A.append(out["prior_mean"].float().cpu())
        if hierarchical:
            S1.append(out["s1"].float().cpu())
            S2.append(out["s2"].float().cpu())
        Y.append(batch["presence"].cpu())
        G.append(batch["group_id"].cpu())
        NV.append(batch["n_views"].cpu())
    out = {"H": torch.cat(H).numpy().astype(np.float64),
           "offset": torch.cat(A).numpy().astype(np.float64),
           "D": torch.cat(Y).numpy().astype(np.float64),
           "group_id": torch.cat(G).numpy(),
           "n_views": torch.cat(NV).numpy()}
    out["s1"] = torch.cat(S1).numpy().astype(np.float64) if S1 else np.zeros_like(out["D"])
    out["s2"] = torch.cat(S2).numpy().astype(np.float64) if S2 else np.zeros_like(out["D"])
    return out


def run_stage2(cfg: Config, run_dir: Path, tag: str = "M3") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(run_dir.glob("*_best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no *_best.pt checkpoint in {run_dir}")
    ck = torch.load(ckpts[0], map_location=device, weights_only=False)
    presence, counts, row_of, pres_names, cnt_names = load_targets(cfg)
    cfg.model.n_presence, cfg.model.n_count = presence.shape[1], counts.shape[1]
    model = DeepBayesFrag(cfg.model).to(device)
    model.load_state_dict(ck["model"])

    store = ViewStore(cfg.paths.views, "train")
    splits = split_molecules(cfg, cfg.data.exclude_oracle_from_training)
    common = dict(presence=presence, counts=counts, target_row=row_of,
                  max_peaks=cfg.data.max_peaks or 0, train=False,
                  max_views=cfg.train.views_per_molecule, seed=cfg.train.seed)

    rng = np.random.RandomState(cfg.bayes.seed)
    fit_ids = splits["train"]
    if cfg.bayes.subsample_molecules and len(fit_ids) > cfg.bayes.subsample_molecules:
        fit_ids = rng.choice(fit_ids, cfg.bayes.subsample_molecules, replace=False)

    def loader_for(ids):
        ds = MoleculeViewDataset(store, ids, **common)
        return DataLoader(ds, batch_size=cfg.train.batch_molecules, shuffle=False,
                          collate_fn=collate, num_workers=cfg.train.num_workers)

    print(f"[stage2] extracting statistics for {len(fit_ids)} fit and "
          f"{len(splits['test'])} evaluation molecules", flush=True)
    fit = extract_statistics(model, loader_for(fit_ids), device)
    test = extract_statistics(model, loader_for(splits["test"]), device)

    # restrict to targets carrying enough posterior mass, specification 5.6 and 21.2
    dic = pd.read_csv(cfg.paths.target_dict)
    prev = dic.set_index("variable").loc[pres_names, "fit_prevalence"].to_numpy()
    keep = np.where(prev >= cfg.data.min_prevalence)[0]
    if cfg.bayes.max_targets:
        keep = keep[np.argsort(-prev[keep])[: cfg.bayes.max_targets]]
    print(f"[stage2] {len(keep)} of {len(pres_names)} targets above prevalence "
          f"{cfg.data.min_prevalence}; polyagamma={HAVE_POLYAGAMMA}", flush=True)

    sampler = PGBARTSampler(
        fit["H"], fit["D"][:, keep], fit["offset"][:, keep],
        fit["s1"][:, keep], fit["s2"][:, keep],
        n_factors=cfg.bayes.n_factors, n_trees=cfg.bayes.n_trees,
        alpha=cfg.bayes.alpha_tree, beta=cfg.bayes.beta_tree,
        leaf_sd=cfg.bayes.leaf_prior_sd, global_scale=cfg.bayes.horseshoe_global_scale,
        seed=cfg.bayes.seed)
    draws = sampler.run(cfg.bayes.n_iter, cfg.bayes.n_burn, cfg.bayes.thin)

    # Held-out molecules were not in the sampler, so the residual is applied through
    # the fitted trees evaluated at their representations.
    res_test = np.zeros((draws["residual_mean"].shape[0], test["H"].shape[0], len(keep)),
                        dtype=np.float32)
    for r, ens in enumerate(sampler.ensembles):
        fx = _apply_forest(ens, sampler.H, test["H"])
        for s in range(res_test.shape[0]):
            res_test[s] += np.outer(fx, draws["loadings"][s][:, r]).astype(np.float32)
    test_draws = {"residual_mean": res_test, "tau": draws["tau"],
                  "loadings": draws["loadings"]}

    p_m3 = posterior_presence(test_draws, test["offset"][:, keep],
                              test["s1"][:, keep], test["s2"][:, keep])
    var = 1.0 / (np.exp(-2.0 * model.aggregator.log_tau.detach().cpu().numpy()[keep])
                 + test["s2"][:, keep]) if isinstance(model.aggregator, MeasurementModel) \
        else np.zeros_like(p_m3)
    mean_m2 = (test["offset"][:, keep] * np.exp(
        -2.0 * model.aggregator.log_tau.detach().cpu().numpy()[keep])
        + test["s1"][:, keep]) * var if isinstance(model.aggregator, MeasurementModel) \
        else test["offset"][:, keep]
    p_m2 = 1.0 / (1.0 + np.exp(-mean_m2 / np.sqrt(1.0 + np.pi * var / 8.0)))

    y = test["D"][:, keep]
    rep = {
        "tag": tag, "n_targets": int(len(keep)), "n_fit_molecules": int(len(fit_ids)),
        "n_eval_molecules": int(y.shape[0]), "polyagamma": HAVE_POLYAGAMMA,
        "M2_deep_only": ev.presence_metrics(p_m2, y),
        "M3_with_bart": ev.presence_metrics(p_m3, y),
    }
    strata = {"views_used": test["n_views"]}
    rep["M2_stratified"] = ev.stratified_calibration(p_m2, y, strata)
    rep["M3_stratified"] = ev.stratified_calibration(p_m3, y, strata)
    np.savez_compressed(run_dir / f"{tag}_bayes.npz", p_m2=p_m2, p_m3=p_m3, y=y,
                        keep=keep, n_views=test["n_views"])
    (run_dir / f"{tag}_bayes_report.json").write_text(json.dumps(rep, indent=2),
                                                      encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if "stratified" not in k}, indent=2),
          flush=True)
    return rep


def _apply_forest(ensemble, X_fit: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    """Evaluate a fitted sum-of-trees at new design points."""
    out = np.zeros(X_new.shape[0])
    for tree in ensemble.trees:
        node = np.zeros(X_new.shape[0], dtype=np.int64)
        active = np.ones(X_new.shape[0], dtype=bool)
        while active.any():
            moved = False
            for n in np.unique(node[active]):
                if tree.is_leaf[n]:
                    continue
                m = active & (node == n)
                go_left = X_new[m, tree.feature[n]] <= tree.threshold[n]
                nxt = np.where(go_left, tree.left[n], tree.right[n])
                node[m] = nxt
                moved = True
            if not moved:
                break
        for n in np.unique(node):
            out[node == n] += tree.value[n]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--run", type=Path, required=True, help="directory holding *_best.pt")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--burn", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--max-targets", type=int, default=None)
    a = ap.parse_args()
    cfg = Config.ablation("M3", a.root)
    if a.iter: cfg.bayes.n_iter = a.iter
    if a.burn: cfg.bayes.n_burn = a.burn
    if a.subsample: cfg.bayes.subsample_molecules = a.subsample
    if a.max_targets: cfg.bayes.max_targets = a.max_targets
    run_stage2(cfg, a.run)


if __name__ == "__main__":
    main()
